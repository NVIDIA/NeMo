#ifndef CUFFTDX_FFT_8_FP64_FWD_PTX_HPP
#define CUFFTDX_FFT_8_FP64_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<424, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .f64 fd<90>;
.reg .b64 rd<2>;
add.f64 fd33, %16, %26;
add.f64 fd34, %17, %28;
sub.f64 fd35, %16, %26;
sub.f64 fd36, %17, %28;
add.f64 fd37, %21, %32;
add.f64 fd38, %23, %33;
sub.f64 fd39, %21, %32;
sub.f64 fd40, %23, %33;
add.f64 fd41, fd33, fd37;
add.f64 fd42, fd34, fd38;
sub.f64 fd43, fd33, fd37;
sub.f64 fd44, fd34, fd38;
add.f64 fd45, fd35, fd40;
sub.f64 fd46, fd36, fd39;
sub.f64 fd47, fd35, fd40;
add.f64 fd48, fd36, fd39;
add.f64 fd49, %18, %29;
add.f64 fd50, %20, %31;
sub.f64 fd51, %18, %29;
sub.f64 fd52, %20, %31;
add.f64 fd53, %24, %34;
add.f64 fd54, %25, %35;
sub.f64 fd55, %24, %34;
sub.f64 fd56, %25, %35;
add.f64 fd57, fd49, fd53;
add.f64 fd58, fd50, fd54;
sub.f64 fd59, fd49, fd53;
sub.f64 fd60, fd50, fd54;
add.f64 fd61, fd51, fd56;
sub.f64 fd62, fd52, fd55;
sub.f64 fd63, fd51, fd56;
add.f64 fd64, fd52, fd55;
mul.f64 fd65, fd61, 0d3FE6A09E667F3BCD;
mul.f64 fd66, fd62, 0dBFE6A09E667F3BCD;
sub.f64 fd67, fd65, fd66;
mul.f64 fd68, fd62, 0d3FE6A09E667F3BCD;
fma.rn.f64 fd69, fd61, 0dBFE6A09E667F3BCD, fd68;
mul.f64 fd70, fd63, 0dBFE6A09E667F3BCD;
mul.f64 fd71, fd64, 0dBFE6A09E667F3BCD;
sub.f64 fd72, fd70, fd71;
add.f64 fd73, fd70, fd71;
add.f64 %1, fd42, fd58;
add.f64 %0, fd41, fd57;
add.f64 %3, fd46, fd69;
add.f64 %2, fd45, fd67;
sub.f64 %5, fd44, fd59;
add.f64 %4, fd43, fd60;
add.f64 %7, fd48, fd73;
add.f64 %6, fd47, fd72;
sub.f64 %9, fd42, fd58;
sub.f64 %8, fd41, fd57;
sub.f64 %11, fd46, fd69;
sub.f64 %10, fd45, fd67;
add.f64 %13, fd44, fd59;
sub.f64 %12, fd43, fd60;
sub.f64 %15, fd48, fd73;
sub.f64 %14, fd47, fd72;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y), "=d"(rmem[4].x), "=d"(rmem[4].y), "=d"(rmem[5].x), "=d"(rmem[5].y), "=d"(rmem[6].x), "=d"(rmem[6].y), "=d"(rmem[7].x), "=d"(rmem[7].y): "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y), "d"(rmem[4].x), "d"(rmem[4].y), "d"(rmem[4].y), "d"(rmem[5].x), "d"(rmem[5].y), "d"(rmem[5].y), "d"(rmem[6].x), "d"(rmem[6].y), "d"(rmem[7].x), "d"(rmem[7].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<425, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<14>;
.reg .f64 fd<77>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 6;
mov.u32 r3, %8;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f64 fd17, %10, %15;
add.f64 fd18, %11, %17;
sub.f64 fd19, %10, %15;
sub.f64 fd20, %11, %17;
add.f64 fd21, %12, %18;
add.f64 fd22, %14, %19;
sub.f64 fd23, %12, %18;
sub.f64 fd24, %14, %19;
add.f64 fd25, fd17, fd21;
add.f64 fd26, fd18, fd22;
sub.f64 fd27, fd17, fd21;
sub.f64 fd28, fd18, fd22;
add.f64 fd29, fd19, fd24;
sub.f64 fd30, fd20, fd23;
sub.f64 fd31, fd19, fd24;
add.f64 fd32, fd20, fd23;
and.b32 r6, r5, 1;
shl.b32 r7, r5, 4;
cvt.u64.u32 rd2, r7;
and.b64 rd3, rd2, 16;
mov.u64 rd4, %9;
add.s64 rd5, rd4, rd3;
ld.global.v2.f64 {fd33, fd34}, [rd5];
mul.f64 fd37, fd33, fd29;
mul.f64 fd38, fd34, fd30;
sub.f64 fd39, fd37, fd38;
mul.f64 fd40, fd33, fd30;
fma.rn.f64 fd41, fd34, fd29, fd40;
mul.f64 fd42, fd33, fd33;
mul.f64 fd43, fd34, fd34;
sub.f64 fd44, fd42, fd43;
mul.f64 fd45, fd34, fd33;
fma.rn.f64 fd46, fd34, fd33, fd45;
mul.f64 fd47, fd44, fd27;
mul.f64 fd48, fd46, fd28;
sub.f64 fd49, fd47, fd48;
mul.f64 fd50, fd44, fd28;
fma.rn.f64 fd51, fd46, fd27, fd50;
ld.global.v2.f64 {fd52, fd53}, [rd5+32];
mul.f64 fd56, fd52, fd31;
mul.f64 fd57, fd53, fd32;
sub.f64 fd58, fd56, fd57;
mul.f64 fd59, fd52, fd32;
fma.rn.f64 fd60, fd53, fd31, fd59;
shl.b32 r8, r5, 5;
and.b32 r9, r8, -64;
add.s32 r10, r4, r9;
barrier.sync 0;
and.b32 r11, r8, 32;
add.s32 r12, r10, r11;
st.shared.v2.f64 [r12], {fd25, fd39};
st.shared.v2.f64 [r12+16], {fd49, fd58};
barrier.sync 0;
mad.lo.s32 r13, r6, -24, r12;
ld.shared.f64 fd61, [r13];
ld.shared.f64 fd62, [r13+16];
ld.shared.f64 fd63, [r13+32];
ld.shared.f64 fd64, [r13+48];
barrier.sync 0;
st.shared.v2.f64 [r12], {fd26, fd41};
st.shared.v2.f64 [r12+16], {fd51, fd60};
barrier.sync 0;
ld.shared.f64 fd65, [r13];
ld.shared.f64 fd66, [r13+16];
ld.shared.f64 fd67, [r13+32];
ld.shared.f64 fd68, [r13+48];
add.f64 %0, fd61, fd63;
add.f64 %1, fd65, fd67;
add.f64 %2, fd62, fd64;
add.f64 %3, fd66, fd68;
sub.f64 %4, fd61, fd63;
sub.f64 %5, fd65, fd67;
sub.f64 %6, fd62, fd64;
sub.f64 %7, fd66, fd68;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y): "r"(smem), "l"(lut_dp_4_8), "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<426, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<21>;
.reg .f64 fd<47>;
.reg .b64 rd<9>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 6;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f64 fd9, %7, %9;
add.f64 fd10, %8, %10;
sub.f64 fd11, %7, %9;
sub.f64 fd12, %8, %10;
shl.b32 r6, r5, 4;
cvt.u64.u32 rd2, r6;
and.b64 rd3, rd2, 48;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f64 {fd13, fd14}, [rd5];
mul.f64 fd17, fd13, fd11;
mul.f64 fd18, fd14, fd12;
sub.f64 fd19, fd17, fd18;
mul.f64 fd20, fd13, fd12;
fma.rn.f64 fd21, fd14, fd11, fd20;
and.b32 r7, r6, -64;
add.s32 r8, r4, r7;
barrier.sync 0;
and.b32 r9, r6, 48;
add.s32 r10, r8, r9;
st.shared.v2.f64 [r10], {fd9, fd19};
barrier.sync 0;
shl.b32 r11, r5, 3;
and.b32 r12, r11, 24;
sub.s32 r13, r10, r12;
ld.shared.f64 fd22, [r13];
ld.shared.f64 fd23, [r13+32];
barrier.sync 0;
st.shared.v2.f64 [r10], {fd10, fd21};
barrier.sync 0;
ld.shared.f64 fd24, [r13];
ld.shared.f64 fd25, [r13+32];
add.f64 fd26, fd22, fd23;
add.f64 fd27, fd24, fd25;
sub.f64 fd28, fd22, fd23;
sub.f64 fd29, fd24, fd25;
bfe.u32 r14, r5, 1, 1;
mul.wide.u32 rd6, r14, 16;
mov.u64 rd7, %6;
add.s64 rd8, rd7, rd6;
ld.global.v2.f64 {fd30, fd31}, [rd8];
mul.f64 fd34, fd30, fd28;
mul.f64 fd35, fd31, fd29;
sub.f64 fd36, fd34, fd35;
mul.f64 fd37, fd30, fd29;
fma.rn.f64 fd38, fd31, fd28, fd37;
and.b32 r15, r11, 8;
add.s32 r16, r8, r15;
barrier.sync 0;
and.b32 r17, r6, 32;
add.s32 r18, r16, r17;
st.shared.f64 [r18], fd26;
st.shared.f64 [r18+16], fd36;
barrier.sync 0;
and.b32 r19, r11, 16;
sub.s32 r20, r18, r19;
ld.shared.f64 fd39, [r20];
ld.shared.f64 fd40, [r20+32];
barrier.sync 0;
st.shared.f64 [r18], fd27;
st.shared.f64 [r18+16], fd38;
barrier.sync 0;
ld.shared.f64 fd41, [r20];
ld.shared.f64 fd42, [r20+32];
add.f64 %0, fd39, fd40;
add.f64 %1, fd41, fd42;
sub.f64 %2, fd39, fd40;
sub.f64 %3, fd41, fd42;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y): "r"(smem), "l"(lut_dp_2_8), "l"(lut_dp_2_4), "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<427, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<14>;
.reg .f64 fd<85>;
.reg .b64 rd<6>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 7;
mov.u32 r3, %8;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
add.f64 fd17, %10, %15;
add.f64 fd18, %11, %17;
sub.f64 fd19, %10, %15;
sub.f64 fd20, %11, %17;
add.f64 fd21, %12, %18;
add.f64 fd22, %14, %19;
sub.f64 fd23, %12, %18;
sub.f64 fd24, %14, %19;
sub.f64 fd25, fd17, fd21;
sub.f64 fd26, fd18, fd22;
add.f64 fd27, fd19, fd24;
sub.f64 fd28, fd20, fd23;
sub.f64 fd29, fd19, fd24;
add.f64 fd30, fd20, fd23;
and.b32 r6, r5, 1;
shl.b32 r7, r5, 6;
and.b32 r8, r7, -128;
add.s32 r9, r4, r8;
shl.b32 r10, r5, 4;
cvt.u64.u32 rd2, r10;
and.b64 rd3, rd2, 16;
mov.u64 rd4, %9;
add.s64 rd5, rd4, rd3;
ld.global.v2.f64 {fd31, fd32}, [rd5];
mul.f64 fd35, fd31, fd27;
mul.f64 fd36, fd32, fd28;
mul.f64 fd37, fd31, fd28;
mul.f64 fd38, fd31, fd31;
mul.f64 fd39, fd32, fd32;
sub.f64 fd40, fd38, fd39;
mul.f64 fd41, fd32, fd31;
fma.rn.f64 fd42, fd32, fd31, fd41;
mul.f64 fd43, fd40, fd25;
mul.f64 fd44, fd42, fd26;
mul.f64 fd45, fd40, fd26;
ld.global.v2.f64 {fd46, fd47}, [rd5+32];
mul.f64 fd50, fd46, fd29;
mul.f64 fd51, fd47, fd30;
mul.f64 fd52, fd46, fd30;
barrier.sync 0;
and.b32 r11, r7, 64;
add.s32 r12, r9, r11;
add.f64 fd53, fd18, fd22;
add.f64 fd54, fd17, fd21;
st.shared.v2.f64 [r12], {fd54, fd53};
fma.rn.f64 fd55, fd32, fd27, fd37;
sub.f64 fd56, fd35, fd36;
st.shared.v2.f64 [r12+16], {fd56, fd55};
sub.f64 fd57, fd43, fd44;
fma.rn.f64 fd58, fd42, fd25, fd45;
st.shared.v2.f64 [r12+32], {fd57, fd58};
fma.rn.f64 fd59, fd47, fd29, fd52;
sub.f64 fd60, fd50, fd51;
st.shared.v2.f64 [r12+48], {fd60, fd59};
barrier.sync 0;
mad.lo.s32 r13, r6, -48, r12;
ld.shared.v2.f64 {fd61, fd62}, [r13];
ld.shared.v2.f64 {fd65, fd66}, [r13+32];
ld.shared.v2.f64 {fd69, fd70}, [r13+64];
ld.shared.v2.f64 {fd73, fd74}, [r13+96];
add.f64 %1, fd62, fd70;
add.f64 %0, fd61, fd69;
add.f64 %3, fd66, fd74;
add.f64 %2, fd65, fd73;
sub.f64 %5, fd62, fd70;
sub.f64 %4, fd61, fd69;
sub.f64 %7, fd66, fd74;
sub.f64 %6, fd65, fd73;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y), "=d"(rmem[2].x), "=d"(rmem[2].y), "=d"(rmem[3].x), "=d"(rmem[3].y): "r"(smem), "l"(lut_dp_4_8), "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y), "d"(rmem[1].y), "d"(rmem[2].x), "d"(rmem[2].y), "d"(rmem[2].y), "d"(rmem[3].x), "d"(rmem[3].y));
};




template<> __forceinline__ __device__ void cufftdx_private_function<428, double, 1>(cufftdx::detail::complex<double> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<21>;
.reg .f64 fd<55>;
.reg .b64 rd<9>;
mov.u32 r1, %tid.y;
shl.b32 r2, r1, 7;
mov.u32 r3, %4;
add.s32 r4, r3, r2;
mov.u32 r5, %tid.x;
sub.f64 fd9, %7, %9;
sub.f64 fd10, %8, %10;
shl.b32 r6, r5, 5;
and.b32 r7, r6, -128;
add.s32 r8, r4, r7;
shl.b32 r9, r5, 4;
cvt.u64.u32 rd2, r9;
and.b64 rd3, rd2, 48;
mov.u64 rd4, %5;
add.s64 rd5, rd4, rd3;
ld.global.v2.f64 {fd11, fd12}, [rd5];
mul.f64 fd15, fd11, fd9;
mul.f64 fd16, fd12, fd10;
mul.f64 fd17, fd11, fd10;
barrier.sync 0;
and.b32 r10, r6, 96;
add.s32 r11, r8, r10;
add.f64 fd18, %8, %10;
add.f64 fd19, %7, %9;
st.shared.v2.f64 [r11], {fd19, fd18};
sub.f64 fd20, fd15, fd16;
fma.rn.f64 fd21, fd12, fd9, fd17;
st.shared.v2.f64 [r11+16], {fd20, fd21};
barrier.sync 0;
and.b32 r12, r9, 48;
sub.s32 r13, r11, r12;
ld.shared.v2.f64 {fd22, fd23}, [r13];
ld.shared.v2.f64 {fd26, fd27}, [r13+64];
sub.f64 fd30, fd22, fd26;
sub.f64 fd31, fd23, fd27;
bfe.u32 r14, r5, 1, 1;
mul.wide.u32 rd6, r14, 16;
mov.u64 rd7, %6;
add.s64 rd8, rd7, rd6;
ld.global.v2.f64 {fd32, fd33}, [rd8];
mul.f64 fd36, fd32, fd30;
mul.f64 fd37, fd33, fd31;
mul.f64 fd38, fd32, fd31;
and.b32 r15, r9, 16;
add.s32 r16, r8, r15;
barrier.sync 0;
and.b32 r17, r6, 64;
add.s32 r18, r16, r17;
add.f64 fd39, fd23, fd27;
add.f64 fd40, fd22, fd26;
st.shared.v2.f64 [r18], {fd40, fd39};
fma.rn.f64 fd41, fd33, fd30, fd38;
sub.f64 fd42, fd36, fd37;
st.shared.v2.f64 [r18+32], {fd42, fd41};
barrier.sync 0;
and.b32 r19, r9, 32;
sub.s32 r20, r18, r19;
ld.shared.v2.f64 {fd43, fd44}, [r20];
ld.shared.v2.f64 {fd47, fd48}, [r20+64];
add.f64 %1, fd44, fd48;
add.f64 %0, fd43, fd47;
sub.f64 %3, fd44, fd48;
sub.f64 %2, fd43, fd47;
})"
     : "=d"(rmem[0].x), "=d"(rmem[0].y), "=d"(rmem[1].x), "=d"(rmem[1].y): "r"(smem), "l"(lut_dp_2_8), "l"(lut_dp_2_4), "d"(rmem[0].x), "d"(rmem[0].y), "d"(rmem[1].x), "d"(rmem[1].y));
};


#endif
