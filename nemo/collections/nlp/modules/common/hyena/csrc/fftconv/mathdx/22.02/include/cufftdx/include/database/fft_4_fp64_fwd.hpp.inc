#ifndef CUFFTDX_FFT_4_FP64_FWD_PTX_HPP
#define CUFFTDX_FFT_4_FP64_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<421, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<33>;
.reg .b64 rd<2>;
add.f64 fd17, %8, %13;
add.f64 fd18, %9, %15;
sub.f64 fd19, %8, %13;
sub.f64 fd20, %9, %15;
add.f64 fd21, %10, %16;
add.f64 fd22, %12, %17;
sub.f64 fd23, %10, %16;
sub.f64 fd24, %12, %17;
add.f64 %1, fd18, fd22;
add.f64 %0, fd17, fd21;
sub.f64 %3, fd20, fd23;
add.f64 %2, fd19, fd24;
sub.f64 %5, fd18, fd22;
sub.f64 %4, fd17, fd21;
add.f64 %7, fd20, fd23;
sub.f64 %6, fd19, fd24;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<422, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<14>;
.reg .f64 fd<30>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 5;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f64 fd9, %6, %8;
add.f64 fd10, %7, %9;
sub.f64 fd11, %6, %8;
sub.f64 fd12, %7, %9;
shl.b32 r6, r5, 4;
cvt.u64.u32 rd2, r6;
and.b64 rd3, rd2, 16;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f64 {fd13, fd14}, [rd5];
mul.f64 fd17, fd13, fd11;
mul.f64 fd18, fd14, fd12;
sub.f64 fd19, fd17, fd18;
mul.f64 fd20, fd13, fd12;
fma.rn.f64 fd21, fd14, fd11, fd20;
and.b32 r7, r6, -32;
add.s32 r8, r4, r7;
barrier.sync 0;
and.b32 r9, r6, 16;
add.s32 r10, r8, r9;
st.shared.v2.f64 [r10], {fd9, fd19};
barrier.sync 0;
shl.b32 r11, r5, 3;
and.b32 r12, r11, 8;
sub.s32 r13, r10, r12;
ld.shared.f64 fd22, [r13];
ld.shared.f64 fd23, [r13+16];
barrier.sync 0;
st.shared.v2.f64 [r10], {fd10, fd21};
barrier.sync 0;
ld.shared.f64 fd24, [r13];
ld.shared.f64 fd25, [r13+16];
add.f64 %0, fd22, fd23;
add.f64 %1, fd24, fd25;
sub.f64 %2, fd22, fd23;
sub.f64 %3, fd24, fd25;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y): "r"(smem), "l"(lut_dp_2_4), "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<423, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<14>;
.reg .f64 fd<34>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 6;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
sub.f64 fd9, %6, %8;
sub.f64 fd10, %7, %9;
shl.b32 r6, r5, 5;
and.b32 r7, r6, -64;
add.s32 r8, r4, r7;
shl.b32 r9, r5, 4;
cvt.u64.u32 rd2, r9;
and.b64 rd3, rd2, 16;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f64 {fd11, fd12}, [rd5];
mul.f64 fd15, fd11, fd9;
mul.f64 fd16, fd12, fd10;
mul.f64 fd17, fd11, fd10;
barrier.sync 0;
and.b32 r10, r6, 32;
add.s32 r11, r8, r10;
add.f64 fd18, %7, %9;
add.f64 fd19, %6, %8;
st.shared.v2.f64 [r11], {fd19, fd18};
sub.f64 fd20, fd15, fd16;
fma.rn.f64 fd21, fd12, fd9, fd17;
st.shared.v2.f64 [r11+16], {fd20, fd21};
barrier.sync 0;
and.b32 r12, r9, 16;
sub.s32 r13, r11, r12;
ld.shared.v2.f64 {fd22, fd23}, [r13];
ld.shared.v2.f64 {fd26, fd27}, [r13+32];
add.f64 %1, fd23, fd27;
add.f64 %0, fd22, fd26;
sub.f64 %3, fd23, fd27;
sub.f64 %2, fd22, fd26;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y): "r"(smem), "l"(lut_dp_2_4), "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y));
};


#endif
