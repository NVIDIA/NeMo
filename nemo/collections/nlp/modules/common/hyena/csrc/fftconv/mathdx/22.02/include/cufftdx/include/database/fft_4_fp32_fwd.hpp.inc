#ifndef CUFFTDX_FFT_4_FP32_FWD_PTX_HPP
#define CUFFTDX_FFT_4_FP32_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<17, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<33>;
.reg .b64 rd<2>;
add.f32 f17, %8, %13;
add.f32 f18, %9, %15;
sub.f32 f19, %8, %13;
sub.f32 f20, %9, %15;
add.f32 f21, %10, %16;
add.f32 f22, %12, %17;
sub.f32 f23, %10, %16;
sub.f32 f24, %12, %17;
add.f32 %1, f18, f22;
add.f32 %0, f17, f21;
sub.f32 %3, f20, f23;
add.f32 %2, f19, f24;
sub.f32 %5, f18, f22;
sub.f32 %4, f17, f21;
add.f32 %7, f20, f23;
sub.f32 %6, f19, f24;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<18, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<30>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 4;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f32 f9, %6, %8;
add.f32 f10, %7, %9;
sub.f32 f11, %6, %8;
sub.f32 f12, %7, %9;
shl.b32 r6, r5, 3;
cvt.u64.u32 rd2, r6;
and.b64 rd3, rd2, 8;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f13, f14}, [rd5];
mul.f32 f17, f13, f11;
mul.f32 f18, f14, f12;
sub.f32 f19, f17, f18;
mul.f32 f20, f13, f12;
fma.rn.f32 f21, f14, f11, f20;
and.b32 r7, r6, -16;
add.s32 r8, r4, r7;
barrier.sync 0;
and.b32 r9, r6, 8;
add.s32 r10, r8, r9;
st.shared.v2.f32 [r10], {f9, f19};
barrier.sync 0;
shl.b32 r11, r5, 2;
and.b32 r12, r11, 4;
sub.s32 r13, r10, r12;
ld.shared.f32 f22, [r13];
ld.shared.f32 f23, [r13+8];
barrier.sync 0;
st.shared.v2.f32 [r10], {f10, f21};
barrier.sync 0;
ld.shared.f32 f24, [r13];
ld.shared.f32 f25, [r13+8];
add.f32 %0, f22, f23;
add.f32 %1, f24, f25;
sub.f32 %2, f22, f23;
sub.f32 %3, f24, f25;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y): "r"(smem), "l"(lut_sp_2_4), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<19, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<34>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 5;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
sub.f32 f9, %6, %8;
sub.f32 f10, %7, %9;
shl.b32 r6, r5, 4;
and.b32 r7, r6, -32;
add.s32 r8, r4, r7;
shl.b32 r9, r5, 3;
cvt.u64.u32 rd2, r9;
and.b64 rd3, rd2, 8;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f11, f12}, [rd5];
mul.f32 f15, f11, f9;
mul.f32 f16, f12, f10;
mul.f32 f17, f11, f10;
barrier.sync 0;
and.b32 r10, r6, 16;
add.s32 r11, r8, r10;
add.f32 f18, %7, %9;
add.f32 f19, %6, %8;
st.shared.v2.f32 [r11], {f19, f18};
sub.f32 f20, f15, f16;
fma.rn.f32 f21, f12, f9, f17;
st.shared.v2.f32 [r11+8], {f20, f21};
barrier.sync 0;
and.b32 r12, r9, 8;
sub.s32 r13, r11, r12;
ld.shared.v2.f32 {f22, f23}, [r13];
ld.shared.v2.f32 {f26, f27}, [r13+16];
add.f32 %1, f23, f27;
add.f32 %0, f22, f26;
sub.f32 %3, f23, f27;
sub.f32 %2, f22, f26;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y): "r"(smem), "l"(lut_sp_2_4), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y));
};


#endif
