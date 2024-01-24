#include <vector>
#include <utility>
#include <cmath>
#include <torch/extension.h>

#include <cuda/std/complex>
#include <cuda_fp16.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(INTYPE, OUTTYPE, NAME, ...)                     \
  if (INTYPE == at::ScalarType::Half) {                                                  \
    using input_t = at::Half;                                                            \
    using output_t = at::Half;                                                           \
    __VA_ARGS__();                                                                       \
  } else if (INTYPE == at::ScalarType::BFloat16) {                                       \
    using input_t = at::BFloat16;                                                        \
    using output_t = at::BFloat16;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INTYPE == at::ScalarType::Float) && (OUTTYPE == at::ScalarType::Float))  { \
    using input_t = float;                                                               \
    using output_t = float;                                                              \
    __VA_ARGS__();                                                                       \
  } else if ((INTYPE == at::ScalarType::Float) && (OUTTYPE == at::ScalarType::Half))  {  \
    using input_t = float;                                                               \
    using output_t = at::Half;                                                           \
    __VA_ARGS__();                                                                       \
  } else {                                                                               \
    AT_ERROR(#NAME, " not implemented for in-type '", toString(INTYPE), "' and out-type '", toString(OUTTYPE), "'"); \
  }

template <typename input_t, typename output_t=input_t>
void fftconv_fwd_cuda_dispatch(
    const input_t *u, const c10::complex<float> *filter,
    const input_t *v, int head_dim, const input_t *q,
    const float *D, const float *dropout_mask, output_t *out,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size, bool output_hbl_layout, bool fftfp16);

template <typename input_t, typename output_t=input_t>
void fftconv_bwd_cuda_dispatch(
    const output_t *dout,
    const input_t *u, const c10::complex<float> *filter,
    const input_t *v, int head_dim, const input_t *q,
    const float *D, const float *dropout_mask,
    input_t *du, c10::complex<float> *dfilter, float *dD,
    float *dv, input_t *dq,
    bool gelu, bool gelu_inp, bool gelu_q, int batch_size, int H, int signal_size,
    size_t batch_stride, size_t H_stride, int fft_size,
    bool output_hbl_layout, bool fftfp16);

torch::Tensor fftconv_fwd(torch::Tensor u, torch::Tensor filter,
                          torch::Tensor D,
                          c10::optional<torch::Tensor> v, int head_dim, 
                          c10::optional<torch::Tensor> q,
                          c10::optional<torch::Tensor> dropout_mask,
                          bool gelu, bool gelu_inp, bool gelu_q, int fft_size,
                          bool force_fp16_output, bool output_hbl_layout,
                          bool fftfp16
                          ) {
    CHECK_DEVICE(u);
    CHECK_DEVICE(filter);
    CHECK_DEVICE(D);

    TORCH_CHECK(u.stride(-1) == 1);
    TORCH_CHECK(filter.is_contiguous());
    TORCH_CHECK(D.is_contiguous());

    const int batch_size = u.size(0);
    const int H = u.size(1);
    const int L = u.size(2);
    CHECK_SHAPE(u, batch_size, H, L);
    CHECK_SHAPE(filter, H / head_dim, fft_size / 2 + 1);
    CHECK_SHAPE(D, H / head_dim);

    TORCH_CHECK(u.dtype() == torch::kFloat16 || u.dtype() == torch::kFloat32 || u.dtype() == torch::kBFloat16);
    // TODO: check filter.dtype is complex64 (no complex32)
    TORCH_CHECK(D.dtype() == torch::kFloat32);

    if (dropout_mask.has_value()) {
        auto dropout_mask_value = dropout_mask.value();
        CHECK_DEVICE(dropout_mask_value);
        CHECK_SHAPE(dropout_mask_value, batch_size, H);
        TORCH_CHECK(dropout_mask_value.dtype() == torch::kFloat32);
    }
    if (v.has_value()) {
        auto v_value = v.value();
        CHECK_DEVICE(v_value);
        CHECK_SHAPE(v_value, batch_size, H, L);
        TORCH_CHECK(v_value.stride(-1) == 1);
        TORCH_CHECK(v_value.stride(0) == u.stride(0) && v_value.stride(1) == u.stride(1));
        TORCH_CHECK(v_value.dtype() == u.dtype());
    }
    if (q.has_value()) {
        auto q_value = q.value();
        CHECK_DEVICE(q_value);
        CHECK_SHAPE(q_value, batch_size, H, L);
        TORCH_CHECK(q_value.stride(-1) == 1);
        TORCH_CHECK(q_value.stride(0) == u.stride(0) && q_value.stride(1) == u.stride(1));
        TORCH_CHECK(q_value.dtype() == u.dtype());
    }

    TORCH_CHECK((!gelu_inp) && (!gelu_q));
    TORCH_CHECK((H % head_dim) == 0);
    TORCH_CHECK(!fftfp16 || head_dim == 8); // fp16 only suported for head dim 8

    auto opts = u.options();
    at::ScalarType u_dtype = ::detail::scalar_type(u.scalar_type());
    if (u.dtype() == at::ScalarType::BFloat16) { force_fp16_output = false; }
    auto out = !output_hbl_layout
        ? torch::empty({batch_size, H, L}, opts.dtype(force_fp16_output ? torch::kFloat16 : u_dtype))
        : torch::empty({H, batch_size, L}, opts.dtype(force_fp16_output ? torch::kFloat16 : u_dtype)).permute({1, 0, 2});
    TORCH_CHECK((L <= fft_size / 2) && (L % 2 == 0));
    TORCH_CHECK(fft_size >= 16 && fft_size <= 16384 && (fft_size == 1 << int(log2(float(fft_size)))));

    size_t batch_stride = u.stride(0), H_stride = u.stride(1);
    DISPATCH_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), out.scalar_type(), "fftconv_fwd", [&] {
        fftconv_fwd_cuda_dispatch(
            static_cast<input_t *>(u.data_ptr()),
            static_cast<c10::complex<float> *>(filter.data_ptr()),
            v.has_value() ? static_cast<input_t *>(v.value().data_ptr()) : nullptr,
            head_dim,
            q.has_value() ? static_cast<input_t *>(q.value().data_ptr()) : nullptr,
            static_cast<float *>(D.data_ptr()),
            dropout_mask.has_value() ? static_cast<float *>(dropout_mask.value().data_ptr()) : nullptr,
            static_cast<output_t *>(out.data_ptr()),
            gelu, gelu_inp, gelu_q, batch_size, H, L, batch_stride, H_stride, fft_size,
            output_hbl_layout, fftfp16);
    });
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fftconv_bwd(torch::Tensor dout,
            torch::Tensor u,
            torch::Tensor filter,
            torch::Tensor D,
            c10::optional<torch::Tensor> v, int head_dim, 
            c10::optional<torch::Tensor> q,
            c10::optional<torch::Tensor> dropout_mask,
            bool gelu, bool gelu_inp, bool gelu_q, int fft_size,
            bool output_hbl_layout, bool fftfp16) {
    CHECK_DEVICE(dout);
    CHECK_DEVICE(u);
    CHECK_DEVICE(filter);
    CHECK_DEVICE(D);

    TORCH_CHECK(u.stride(-1) == 1);
    TORCH_CHECK(filter.is_contiguous());
    TORCH_CHECK(D.is_contiguous());

    const int batch_size = u.size(0);
    const int H = u.size(1);
    const int L = u.size(2);
    CHECK_SHAPE(dout, batch_size, H, L);
    CHECK_SHAPE(u, batch_size, H, L);
    CHECK_SHAPE(filter, H / head_dim, fft_size / 2 + 1);
    CHECK_SHAPE(D, H / head_dim);
    if (!output_hbl_layout) {
        TORCH_CHECK(dout.is_contiguous());
    } else {
        // Previously we were checking
        // TORCH_CHECK(dout.stride(1) == batch_size * L && dout.stride(0) == L)
        // but this fails for the edge case of batch_size=1, where shape (H, 1, L)
        // is already contiguous, and dout.stride(0) = L * H in that case.
        TORCH_CHECK(dout.permute({1, 0, 2}).is_contiguous());
    }

    TORCH_CHECK(dout.dtype() == torch::kFloat16 || dout.dtype() == torch::kFloat32 || dout.dtype() == torch::kBFloat16);
    TORCH_CHECK(u.dtype() == torch::kFloat16 || u.dtype() == torch::kFloat32 || u.dtype() == torch::kBFloat16);
    TORCH_CHECK(D.dtype() == torch::kFloat32);

    auto opts = u.options();

    torch::Tensor dv;
    torch::Tensor dq;

    if (dropout_mask.has_value()) {
        auto dropout_mask_value = dropout_mask.value();
        CHECK_DEVICE(dropout_mask_value);
        CHECK_SHAPE(dropout_mask_value, batch_size, H);
        TORCH_CHECK(dropout_mask_value.dtype() == torch::kFloat32);
    }
    if (v.has_value()) {
        auto v_value = v.value();
        CHECK_DEVICE(v_value);
        CHECK_SHAPE(v_value, batch_size, H, L);
        TORCH_CHECK(v_value.stride(-1) == 1);
        TORCH_CHECK(v_value.stride(0) == u.stride(0) && v_value.stride(1) == u.stride(1));
        TORCH_CHECK(v_value.dtype() == u.dtype());
        dv = torch::zeros_like(v_value, opts.dtype(torch::kFloat));
    }
    if (q.has_value()) {
        auto q_value = q.value();
        CHECK_DEVICE(q_value);
        CHECK_SHAPE(q_value, batch_size, H, L);
        TORCH_CHECK(q_value.stride(-1) == 1);
        TORCH_CHECK(q_value.stride(0) == u.stride(0) && q_value.stride(1) == u.stride(1));
        TORCH_CHECK(q_value.dtype() == u.dtype());
        dq = torch::empty_like(q_value);
    }

    TORCH_CHECK((!gelu_inp) && (!gelu_q));
    TORCH_CHECK((H % head_dim) == 0);
    TORCH_CHECK(!fftfp16 || head_dim == 8); // fp16 only suported for head dim 8

    auto du = torch::empty_like(u);
    auto dfilter = torch::empty({batch_size, H / head_dim, head_dim, fft_size / 2 + 1}, opts.dtype(filter.dtype()));
    auto dD = torch::empty({batch_size, H / head_dim, head_dim}, opts.dtype(torch::kFloat));

    TORCH_CHECK((L <= fft_size / 2) && (L % 2 == 0));
    TORCH_CHECK(fft_size >= 16 && fft_size <= 16384 && (fft_size == 1 << int(log2(float(fft_size)))));

    size_t batch_stride = u.stride(0), H_stride = u.stride(1);
    DISPATCH_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), dout.scalar_type(), "fftconv_bwd", [&] {
        fftconv_bwd_cuda_dispatch(
            static_cast<output_t *>(dout.data_ptr()),
            static_cast<input_t *>(u.data_ptr()),
            static_cast<c10::complex<float> *>(filter.data_ptr()),
            v.has_value() ? static_cast<input_t *>(v.value().data_ptr()) : nullptr,
            head_dim,
            q.has_value() ? static_cast<input_t *>(q.value().data_ptr()) : nullptr,
            static_cast<float *>(D.data_ptr()),
            dropout_mask.has_value() ? static_cast<float *>(dropout_mask.value().data_ptr()) : nullptr,
            static_cast<input_t *>(du.data_ptr()),
            static_cast<c10::complex<float> *>(dfilter.data_ptr()),
            static_cast<float *>(dD.data_ptr()),
            v.has_value() ? static_cast<float *>(dv.data_ptr()) : nullptr,
            q.has_value() ? static_cast<input_t *>(dq.data_ptr()) : nullptr,
            gelu, gelu_inp, gelu_q, batch_size, H, L, batch_stride, H_stride, fft_size,
            output_hbl_layout, fftfp16);
    });

    return std::make_tuple(du, dfilter.sum(/*dim=*/std::vector<int64_t>{0, 2}), dD.sum(/*dim=*/std::vector<int64_t>{0, 2}), dv, dq);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fftconv_fwd", &fftconv_fwd, "Convolution with FFT");
    m.def("fftconv_bwd", &fftconv_bwd, "Convolution with FFT, backward");
}
