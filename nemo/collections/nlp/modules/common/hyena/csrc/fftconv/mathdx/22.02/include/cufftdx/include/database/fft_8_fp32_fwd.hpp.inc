#ifndef CUFFTDX_FFT_8_FP32_FWD_PTX_HPP
#define CUFFTDX_FFT_8_FP32_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<20, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<90>;
.reg .b64 rd<2>;
add.f32 f33, %16, %26;
add.f32 f34, %17, %28;
sub.f32 f35, %16, %26;
sub.f32 f36, %17, %28;
add.f32 f37, %21, %32;
add.f32 f38, %23, %33;
sub.f32 f39, %21, %32;
sub.f32 f40, %23, %33;
add.f32 f41, f33, f37;
add.f32 f42, f34, f38;
sub.f32 f43, f33, f37;
sub.f32 f44, f34, f38;
add.f32 f45, f35, f40;
sub.f32 f46, f36, f39;
sub.f32 f47, f35, f40;
add.f32 f48, f36, f39;
add.f32 f49, %18, %29;
add.f32 f50, %20, %31;
sub.f32 f51, %18, %29;
sub.f32 f52, %20, %31;
add.f32 f53, %24, %34;
add.f32 f54, %25, %35;
sub.f32 f55, %24, %34;
sub.f32 f56, %25, %35;
add.f32 f57, f49, f53;
add.f32 f58, f50, f54;
sub.f32 f59, f49, f53;
sub.f32 f60, f50, f54;
add.f32 f61, f51, f56;
sub.f32 f62, f52, f55;
sub.f32 f63, f51, f56;
add.f32 f64, f52, f55;
mul.f32 f65, f61, 0f3F3504F3;
mul.f32 f66, f62, 0fBF3504F3;
sub.f32 f67, f65, f66;
mul.f32 f68, f62, 0f3F3504F3;
fma.rn.f32 f69, f61, 0fBF3504F3, f68;
mul.f32 f70, f63, 0fBF3504F3;
mul.f32 f71, f64, 0fBF3504F3;
sub.f32 f72, f70, f71;
add.f32 f73, f70, f71;
add.f32 %1, f42, f58;
add.f32 %0, f41, f57;
add.f32 %3, f46, f69;
add.f32 %2, f45, f67;
sub.f32 %5, f44, f59;
add.f32 %4, f43, f60;
add.f32 %7, f48, f73;
add.f32 %6, f47, f72;
sub.f32 %9, f42, f58;
sub.f32 %8, f41, f57;
sub.f32 %11, f46, f69;
sub.f32 %10, f45, f67;
add.f32 %13, f44, f59;
sub.f32 %12, f43, f60;
sub.f32 %15, f48, f73;
sub.f32 %14, f47, f72;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y), "=f"(rmem[4].x), "=f"(rmem[4].y), "=f"(rmem[5].x), "=f"(rmem[5].y), "=f"(rmem[6].x), "=f"(rmem[6].y), "=f"(rmem[7].x), "=f"(rmem[7].y): "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y), "f"(rmem[4].x), "f"(rmem[4].y), "f"(rmem[4].y), "f"(rmem[5].x), "f"(rmem[5].y), "f"(rmem[5].y), "f"(rmem[6].x), "f"(rmem[6].y), "f"(rmem[7].x), "f"(rmem[7].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<21, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<78>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 5;
mov.u32 r3, %8;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f32 f17, %10, %15;
add.f32 f18, %11, %17;
sub.f32 f19, %10, %15;
sub.f32 f20, %11, %17;
add.f32 f21, %12, %18;
add.f32 f22, %14, %19;
sub.f32 f23, %12, %18;
sub.f32 f24, %14, %19;
add.f32 f25, f17, f21;
add.f32 f26, f18, f22;
sub.f32 f27, f17, f21;
sub.f32 f28, f18, f22;
add.f32 f29, f19, f24;
sub.f32 f30, f20, f23;
sub.f32 f31, f19, f24;
add.f32 f32, f20, f23;
and.b32 r6, r5, 1;
shl.b32 r7, r5, 3;
cvt.u64.u32 rd2, r7;
and.b64 rd3, rd2, 8;
mov.u64 rd4, %9;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f33, f34}, [rd5];
mul.f32 f37, f33, f29;
mul.f32 f38, f34, f30;
sub.f32 f39, f37, f38;
mul.f32 f40, f33, f30;
fma.rn.f32 f41, f34, f29, f40;
mul.f32 f42, f33, f33;
mul.f32 f43, f34, f34;
sub.f32 f44, f42, f43;
mul.f32 f45, f34, f33;
fma.rn.f32 f46, f34, f33, f45;
mul.f32 f47, f44, f27;
mul.f32 f48, f46, f28;
sub.f32 f49, f47, f48;
mul.f32 f50, f44, f28;
fma.rn.f32 f51, f46, f27, f50;
mul.f32 f52, f33, f44;
mul.f32 f53, f34, f46;
sub.f32 f54, f52, f53;
mul.f32 f55, f33, f46;
fma.rn.f32 f56, f34, f44, f55;
mul.f32 f57, f54, f31;
mul.f32 f58, f56, f32;
sub.f32 f59, f57, f58;
mul.f32 f60, f54, f32;
fma.rn.f32 f61, f56, f31, f60;
shl.b32 r8, r5, 4;
and.b32 r9, r8, -32;
add.s32 r10, r4, r9;
barrier.sync 0;
and.b32 r11, r8, 16;
add.s32 r12, r10, r11;
st.shared.v4.f32 [r12], {f25, f39, f49, f59};
barrier.sync 0;
mad.lo.s32 r13, r6, -12, r12;
ld.shared.f32 f62, [r13];
ld.shared.f32 f63, [r13+8];
ld.shared.f32 f64, [r13+16];
ld.shared.f32 f65, [r13+24];
barrier.sync 0;
st.shared.v4.f32 [r12], {f26, f41, f51, f61};
barrier.sync 0;
ld.shared.f32 f66, [r13];
ld.shared.f32 f67, [r13+8];
ld.shared.f32 f68, [r13+16];
ld.shared.f32 f69, [r13+24];
add.f32 %0, f62, f64;
add.f32 %1, f66, f68;
add.f32 %2, f63, f65;
add.f32 %3, f67, f69;
sub.f32 %4, f62, f64;
sub.f32 %5, f66, f68;
sub.f32 %6, f63, f65;
sub.f32 %7, f67, f69;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y): "r"(smem), "l"(lut_sp_4_8), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<22, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<86>;
.reg .b32 r<14>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 6;
mov.u32 r3, %8;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f32 f17, %10, %15;
add.f32 f18, %11, %17;
sub.f32 f19, %10, %15;
sub.f32 f20, %11, %17;
add.f32 f21, %12, %18;
add.f32 f22, %14, %19;
sub.f32 f23, %12, %18;
sub.f32 f24, %14, %19;
sub.f32 f25, f17, f21;
sub.f32 f26, f18, f22;
add.f32 f27, f19, f24;
sub.f32 f28, f20, f23;
sub.f32 f29, f19, f24;
add.f32 f30, f20, f23;
and.b32 r6, r5, 1;
shl.b32 r7, r5, 5;
and.b32 r8, r7, -64;
add.s32 r9, r4, r8;
shl.b32 r10, r5, 3;
cvt.u64.u32 rd2, r10;
and.b64 rd3, rd2, 8;
mov.u64 rd4, %9;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f31, f32}, [rd5];
mul.f32 f35, f31, f27;
mul.f32 f36, f32, f28;
mul.f32 f37, f31, f28;
mul.f32 f38, f31, f31;
mul.f32 f39, f32, f32;
sub.f32 f40, f38, f39;
mul.f32 f41, f32, f31;
fma.rn.f32 f42, f32, f31, f41;
mul.f32 f43, f40, f25;
mul.f32 f44, f42, f26;
mul.f32 f45, f40, f26;
mul.f32 f46, f31, f40;
mul.f32 f47, f32, f42;
sub.f32 f48, f46, f47;
mul.f32 f49, f31, f42;
fma.rn.f32 f50, f32, f40, f49;
mul.f32 f51, f48, f29;
mul.f32 f52, f50, f30;
mul.f32 f53, f48, f30;
barrier.sync 0;
and.b32 r11, r7, 32;
add.s32 r12, r9, r11;
add.f32 f54, f18, f22;
add.f32 f55, f17, f21;
fma.rn.f32 f56, f32, f27, f37;
sub.f32 f57, f35, f36;
st.shared.v4.f32 [r12], {f55, f54, f57, f56};
sub.f32 f58, f43, f44;
fma.rn.f32 f59, f42, f25, f45;
fma.rn.f32 f60, f50, f29, f53;
sub.f32 f61, f51, f52;
st.shared.v4.f32 [r12+16], {f58, f59, f61, f60};
barrier.sync 0;
mad.lo.s32 r13, r6, -24, r12;
ld.shared.v2.f32 {f62, f63}, [r13];
ld.shared.v2.f32 {f66, f67}, [r13+16];
ld.shared.v2.f32 {f70, f71}, [r13+32];
ld.shared.v2.f32 {f74, f75}, [r13+48];
add.f32 %1, f63, f71;
add.f32 %0, f62, f70;
add.f32 %3, f67, f75;
add.f32 %2, f66, f74;
sub.f32 %5, f63, f71;
sub.f32 %4, f62, f70;
sub.f32 %7, f67, f75;
sub.f32 %6, f66, f74;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y), "=f"(rmem[2].x), "=f"(rmem[2].y), "=f"(rmem[3].x), "=f"(rmem[3].y): "r"(smem), "l"(lut_sp_4_8), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y), "f"(rmem[1].y), "f"(rmem[2].x), "f"(rmem[2].y), "f"(rmem[2].y), "f"(rmem[3].x), "f"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<23, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<47>;
.reg .b32 r<21>;
.reg .b64 rd<9>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 5;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f32 f9, %7, %9;
add.f32 f10, %8, %10;
sub.f32 f11, %7, %9;
sub.f32 f12, %8, %10;
shl.b32 r6, r5, 3;
cvt.u64.u32 rd2, r6;
and.b64 rd3, rd2, 24;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f13, f14}, [rd5];
mul.f32 f17, f13, f11;
mul.f32 f18, f14, f12;
sub.f32 f19, f17, f18;
mul.f32 f20, f13, f12;
fma.rn.f32 f21, f14, f11, f20;
and.b32 r7, r6, -32;
add.s32 r8, r4, r7;
barrier.sync 0;
and.b32 r9, r6, 24;
add.s32 r10, r8, r9;
st.shared.v2.f32 [r10], {f9, f19};
barrier.sync 0;
shl.b32 r11, r5, 2;
and.b32 r12, r11, 12;
sub.s32 r13, r10, r12;
ld.shared.f32 f22, [r13];
ld.shared.f32 f23, [r13+16];
barrier.sync 0;
st.shared.v2.f32 [r10], {f10, f21};
barrier.sync 0;
ld.shared.f32 f24, [r13];
ld.shared.f32 f25, [r13+16];
add.f32 f26, f22, f23;
add.f32 f27, f24, f25;
sub.f32 f28, f22, f23;
sub.f32 f29, f24, f25;
bfe.u32 r14, r5, 1, 1;
mul.wide.u32 rd6, r14, 8;
mov.u64 rd7, %6;
add.s64 rd8, rd7, rd6;
ld.global.v2.f32 {f30, f31}, [rd8];
mul.f32 f34, f30, f28;
mul.f32 f35, f31, f29;
sub.f32 f36, f34, f35;
mul.f32 f37, f30, f29;
fma.rn.f32 f38, f31, f28, f37;
and.b32 r15, r11, 4;
add.s32 r16, r8, r15;
barrier.sync 0;
and.b32 r17, r6, 16;
add.s32 r18, r16, r17;
st.shared.f32 [r18], f26;
st.shared.f32 [r18+8], f36;
barrier.sync 0;
and.b32 r19, r11, 8;
sub.s32 r20, r18, r19;
ld.shared.f32 f39, [r20];
ld.shared.f32 f40, [r20+16];
barrier.sync 0;
st.shared.f32 [r18], f27;
st.shared.f32 [r18+8], f38;
barrier.sync 0;
ld.shared.f32 f41, [r20];
ld.shared.f32 f42, [r20+16];
add.f32 %0, f39, f40;
add.f32 %1, f41, f42;
sub.f32 %2, f39, f40;
sub.f32 %3, f41, f42;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y): "r"(smem), "l"(lut_sp_2_8), "l"(lut_sp_2_4), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<24, float, 1>(cufftdx::detail::complex<float> *rmem, unsigned smem){

asm volatile (R"({
.reg .f32 f<55>;
.reg .b32 r<21>;
.reg .b64 rd<9>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 6;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
sub.f32 f9, %7, %9;
sub.f32 f10, %8, %10;
shl.b32 r6, r5, 4;
and.b32 r7, r6, -64;
add.s32 r8, r4, r7;
shl.b32 r9, r5, 3;
cvt.u64.u32 rd2, r9;
and.b64 rd3, rd2, 24;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f32 {f11, f12}, [rd5];
mul.f32 f15, f11, f9;
mul.f32 f16, f12, f10;
mul.f32 f17, f11, f10;
barrier.sync 0;
and.b32 r10, r6, 48;
add.s32 r11, r8, r10;
add.f32 f18, %8, %10;
add.f32 f19, %7, %9;
st.shared.v2.f32 [r11], {f19, f18};
sub.f32 f20, f15, f16;
fma.rn.f32 f21, f12, f9, f17;
st.shared.v2.f32 [r11+8], {f20, f21};
barrier.sync 0;
and.b32 r12, r9, 24;
sub.s32 r13, r11, r12;
ld.shared.v2.f32 {f22, f23}, [r13];
ld.shared.v2.f32 {f26, f27}, [r13+32];
sub.f32 f30, f22, f26;
sub.f32 f31, f23, f27;
bfe.u32 r14, r5, 1, 1;
mul.wide.u32 rd6, r14, 8;
mov.u64 rd7, %6;
add.s64 rd8, rd7, rd6;
ld.global.v2.f32 {f32, f33}, [rd8];
mul.f32 f36, f32, f30;
mul.f32 f37, f33, f31;
mul.f32 f38, f32, f31;
and.b32 r15, r9, 8;
add.s32 r16, r8, r15;
barrier.sync 0;
and.b32 r17, r6, 32;
add.s32 r18, r16, r17;
add.f32 f39, f23, f27;
add.f32 f40, f22, f26;
st.shared.v2.f32 [r18], {f40, f39};
fma.rn.f32 f41, f33, f30, f38;
sub.f32 f42, f36, f37;
st.shared.v2.f32 [r18+16], {f42, f41};
barrier.sync 0;
and.b32 r19, r9, 16;
sub.s32 r20, r18, r19;
ld.shared.v2.f32 {f43, f44}, [r20];
ld.shared.v2.f32 {f47, f48}, [r20+32];
add.f32 %1, f44, f48;
add.f32 %0, f43, f47;
sub.f32 %3, f44, f48;
sub.f32 %2, f43, f47;
})"
     : "=f"(rmem[0].x), "=f"(rmem[0].y), "=f"(rmem[1].x), "=f"(rmem[1].y): "r"(smem), "l"(lut_sp_2_8), "l"(lut_sp_2_4), "f"(rmem[0].x), "f"(rmem[0].y), "f"(rmem[1].x), "f"(rmem[1].y));
};


#endif
