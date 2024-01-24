#ifndef CUFFTDX_FFT_4_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_4_FP16_INV_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<965, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<51>;
.reg .b64 rd<2>;
{
add.f16x2 r1, %8, %9;
}
{
add.f16x2 r4, %10, %11;
}
{
sub.f16x2 r7, %8, %9;
}
{
sub.f16x2 r10, %10, %11;
}
{
add.f16x2 r13, %12, %13;
}
{
add.f16x2 r16, %14, %15;
}
{
sub.f16x2 r19, %12, %13;
}
{
sub.f16x2 r22, %14, %15;
}
{
neg.f16x2 r25, r22;
}
{
add.f16x2 %0, r1, r13;
}
{
add.f16x2 %1, r4, r16;
}
{
sub.f16x2 %4, r1, r13;
}
{
sub.f16x2 %5, r4, r16;
}
{
add.f16x2 %2, r7, r25;
}
{
add.f16x2 %3, r10, r19;
}
{
sub.f16x2 %6, r7, r25;
}
{
sub.f16x2 %7, r10, r19;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[3].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<966, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<10>;
.reg .b32 r<79>;
.reg .b64 rd<2>;
mov.u32 r65, %tid.y;
shl.b32 r66, r65, 4;
mov.u32 r67, %4;
add.s32 r68, r67, r66;
mov.u32 r69, %tid.x;
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
and.b32 r70, r69, 1;
shl.b32 r71, r69, 3;
and.b32 r72, r71, -16;
add.s32 r73, r68, r72;
cvt.rn.f32.u32 f7, r70;
mul.f32 f8, f7, 0f3FC90FDB;
cos.approx.f32 f1, f8;
sin.approx.f32 f9, f8;
neg.f32 f2, f9;
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
fma.rn.f16x2 r23, r7, r16, r20;
}
{
mul.f16x2 r27, r7, r18;
}
{
neg.f16x2 r30, r27;
}
{
fma.rn.f16x2 r32, r10, r16, r30;
}
barrier.sync 0;
and.b32 r74, r71, 8;
add.s32 r75, r73, r74;
st.shared.v2.f32 [r75], {r1, r23};
barrier.sync 0;
shl.b32 r76, r69, 2;
and.b32 r77, r76, 4;
sub.s32 r78, r75, r77;
ld.shared.u32 r54, [r78];
ld.shared.u32 r55, [r78+8];
barrier.sync 0;
st.shared.v2.f32 [r75], {r4, r32};
barrier.sync 0;
ld.shared.u32 r57, [r78];
ld.shared.u32 r58, [r78+8];
{
add.f16x2 %0, r54, r55;
}
{
add.f16x2 %1, r57, r58;
}
{
sub.f16x2 %2, r54, r55;
}
{
sub.f16x2 %3, r57, r58;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};




template<> __forceinline__ __device__ void cufftdx_private_function<967, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<10>;
.reg .b32 r<79>;
.reg .b64 rd<2>;
mov.u32 r65, %tid.y;
shl.b32 r66, r65, 5;
mov.u32 r67, %4;
add.s32 r68, r67, r66;
mov.u32 r69, %tid.x;
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
and.b32 r70, r69, 1;
shl.b32 r71, r69, 4;
and.b32 r72, r71, -32;
add.s32 r73, r68, r72;
cvt.rn.f32.u32 f7, r70;
mul.f32 f8, f7, 0f3FC90FDB;
cos.approx.f32 f1, f8;
sin.approx.f32 f9, f8;
neg.f32 f2, f9;
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
fma.rn.f16x2 r23, r7, r16, r20;
}
{
mul.f16x2 r27, r7, r18;
}
{
neg.f16x2 r30, r27;
}
{
fma.rn.f16x2 r32, r10, r16, r30;
}
barrier.sync 0;
and.b32 r74, r71, 16;
add.s32 r75, r73, r74;
st.shared.v2.f32 [r75], {r1, r4};
st.shared.v2.f32 [r75+8], {r23, r32};
barrier.sync 0;
shl.b32 r76, r69, 3;
and.b32 r77, r76, 8;
sub.s32 r78, r75, r77;
ld.shared.u32 r54, [r78];
ld.shared.u32 r57, [r78+4];
ld.shared.u32 r55, [r78+16];
ld.shared.u32 r58, [r78+20];
{
add.f16x2 %0, r54, r55;
}
{
add.f16x2 %1, r57, r58;
}
{
sub.f16x2 %2, r54, r55;
}
{
sub.f16x2 %3, r57, r58;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(smem), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
