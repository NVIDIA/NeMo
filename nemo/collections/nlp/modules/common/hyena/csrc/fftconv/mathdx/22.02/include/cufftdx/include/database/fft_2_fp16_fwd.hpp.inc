#ifndef CUFFTDX_FFT_2_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_2_FP16_FWD_PTX_HPP



template<> __forceinline__ __device__ void cufftdx_private_function<762, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({
.reg .b32 r<13>;
.reg .b64 rd<2>;
{
add.f16x2 %0, %4, %5;
}
{
add.f16x2 %1, %6, %7;
}
{
sub.f16x2 %2, %4, %5;
}
{
sub.f16x2 %3, %6, %7;
}
})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)): "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[1].y)));
};


#endif
