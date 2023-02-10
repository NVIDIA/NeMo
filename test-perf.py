import nemo
import nemo.collections.asr as nemo_asr
import onnxruntime
import numpy as np
import torch
from contextlib import nullcontext
import time

from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, LogprobsType, NeuralType, SpectrogramType, ChannelType
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.classes.common import typecheck
from collections import OrderedDict


class MyEncDecCTCModelBPE(EncDecCTCModelBPE):
    def __init__(self, *args, **kwargs):
        super(MyEncDecCTCModelBPE, self).__init__(*args, **kwargs)
        self.encoder.export_cache_support = True

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=False),
                "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=False),
                "drop_extra_pre_encoded": NeuralType(tuple('B'), LengthsType(), optional=False),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=False),
                "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=False),
                "drop_extra_pre_encoded_next": NeuralType(tuple('B'), LengthsType(), optional=False),
            }
        )

    def input_example(self, max_batch=1, max_dim=256):
        return self.encoder.input_example(max_batch=max_batch, max_dim=max_dim)

    @typecheck()
    def forward_for_export(self, audio_signal=None, length=None, cache_last_channel=None, cache_last_time=None, drop_extra_pre_encoded=None):
        return self.forward_internal(audio_signal=audio_signal, length=length, cache_last_channel=cache_last_channel, cache_last_time=cache_last_time, drop_extra_pre_encoded=drop_extra_pre_encoded)

    def forward_encoder(self, audio_signal=None, length=None, cache_last_channel=None, cache_last_time=None, drop_extra_pre_encoded=None):
        if self.encoder.streaming_cfg is None:
            self.encoder.setup_streaming_params()

        encoder_output = self.encoder.forward(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )

        if len(encoder_output) == 2:
            encoded, encoded_len = encoder_output
            cache_last_channel_next = cache_last_time_next = None
        else:
            encoded, encoded_len, cache_last_channel_next, cache_last_time_next, drop_extra_pre_encoded_next = encoder_output

        if cache_last_channel_next is not None and self.encoder.streaming_cfg.last_channel_cache_size >= 0:
            if self.encoder.streaming_cfg.last_channel_cache_size > 0:
                cache_last_channel_next = cache_last_channel_next[
                    :, :, -self.encoder.streaming_cfg.last_channel_cache_size :, :
                ]
            else:
                cache_last_channel_next = cache_last_channel_next[:, :, 0:0, :]
        if True:
            encoded = encoded[:, :, : self.encoder.streaming_cfg.valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=self.encoder.streaming_cfg.valid_out_len)

        return encoded, encoded_len, cache_last_channel_next, cache_last_time_next, drop_extra_pre_encoded_next

    def forward_internal(self, audio_signal, length, cache_last_channel, cache_last_time, drop_extra_pre_encoded):
        enc, enc_len, cache_ch_next, cache_time_next, drop_next = self.forward_encoder(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            drop_extra_pre_encoded=drop_extra_pre_encoded)
        #return enc, enc_len, cache_ch_next, cache_time_next
        logits = self.decoder(encoder_output=enc)
        return (logits, enc_len, cache_ch_next, cache_time_next, drop_next)

    @typecheck()
    def forward(self, audio_signal, length, cache_last_channel, cache_last_time, drop_extra_pre_encoded):
        return self.forward_internal(audio_signal=audio_signal, length=length, cache_last_channel=cache_last_channel, cache_last_time=cache_last_time, drop_extra_pre_encoded=drop_extra_pre_encoded)


def infer_nemo(nemo_model, audio, lens, cache_last_channel, cache_last_time, drop_extra_pre_encoded):
    encoded, encoded_len, cache_last_channel_next, cahce_last_time_next = nemo_model.encoder.forward_for_export(processed_signal=audio, processed_signal_length=lens, cache_last_channel=cache_last_channel, cache_last_time=cache_last_time, drop_extra_pre_encoded=drop_extra_pre_encoded)
    log_probs = nemo_model.decoder(encoder_output=encoded)
    return log_probs


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def infer_onnx(model_path, audio, lens, cache_last_channel, cache_last_time, drop_extra_pre_encoded):
    sess = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

    for input in sess.get_inputs():
        print(input.name)

    ort_inputs = {}
    ort_inputs[sess.get_inputs()[0].name] = to_numpy(audio)
    ort_inputs[sess.get_inputs()[1].name] = to_numpy(lens)
    ort_inputs[sess.get_inputs()[2].name] = to_numpy(cache_last_channel)
    ort_inputs[sess.get_inputs()[3].name] = to_numpy(cache_last_time)
    ort_inputs[sess.get_inputs()[4].name] = to_numpy(drop_extra_pre_encoded)

    ort_outs = sess.run(None, ort_inputs)
    return ort_outs[0]

def old_main():
    nemo_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from('/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo')

    nemo_model.export_cache_support = True
    nemo_model.encoder.export_cache_support = True
    audio, lens, cache_last_channel, cache_last_time, drop_extra_pre_encoded = nemo_model.encoder.input_example()

    nemo_log_probs = infer_nemo(nemo_model, audio, lens, cache_last_channel, cache_last_time, drop_extra_pre_encoded)
    onnx_log_probs = infer_onnx('/models/cache-aware-ctc-onnx/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.onnx', audio, lens, cache_last_channel, cache_last_time, drop_extra_pre_encoded)

    assert tuple(nemo_log_probs.size()) == tuple(onnx_log_probs.shape)

    np.testing.assert_allclose(to_numpy(nemo_log_probs), onnx_log_probs, rtol=1e-03, atol=1e-05)
    print("Looks Good")


#def do_save():
#    nemo_model = MyEncDecCTCModelBPE.restore_from('/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo')


def main_enc_dec():
    nemo_model = MyEncDecCTCModelBPE.restore_from('/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo')
    nemo_model.encoder.export_cache_support = True

    autocast = nullcontext
    with autocast(), torch.no_grad(), torch.inference_mode():
        nemo_model.to(device='cuda').freeze()
        nemo_model.eval()
        input_example = nemo_model.input_example(max_batch=32)

        _, desc = nemo_model.export(
            'tmp.onnx',
            input_example=input_example,
            check_trace=False,
            onnx_opset_version=None,
            verbose=False,
        )

        nemo_outs = nemo_model.forward(
            audio_signal=input_example[0],
            length=input_example[1],
            cache_last_channel=input_example[2],
            cache_last_time=input_example[3],
            drop_extra_pre_encoded=input_example[4],
        )

    sess = onnxruntime.InferenceSession('tmp.onnx', providers=['CUDAExecutionProvider'])
    for o in sess.get_outputs():
        print(o.name)
    for o in sess.get_inputs():
        print(o.name)

    ort_inputs = {}
    ort_inputs[sess.get_inputs()[0].name] = to_numpy(input_example[0])
    ort_inputs[sess.get_inputs()[1].name] = to_numpy(input_example[1])
    ort_inputs[sess.get_inputs()[2].name] = to_numpy(input_example[2])
    ort_inputs[sess.get_inputs()[3].name] = to_numpy(input_example[3])
    ort_inputs[sess.get_inputs()[4].name] = to_numpy(input_example[4])

    ort_outs = sess.run(None, ort_inputs)
    print(len(ort_outs))
    for o in ort_outs:
        print(o.shape)

    tol = 0.1

    assert len(ort_outs) == len(nemo_outs)
    all_good = True
    for i in range(len(ort_outs)):
        print(f"Testing {sess.get_outputs()[i]}")
        print(nemo_outs[i].size(), ort_outs[i].shape)
        tout = torch.from_numpy(ort_outs[i])
        if not torch.allclose(tout, nemo_outs[i].cpu(), rtol=tol, atol=100*tol):
            print(f"verification vailed for output {sess.get_outputs()[i]}")
            all_good = False
    if all_good:
        print('Looks Good!')
    else:
        print("Verification failed")

    nemo_greedy = torch.argmax(nemo_outs[0], dim=-1).cpu()
    ort_greedy  = torch.argmax(torch.from_numpy(ort_outs[1]), dim=-1).cpu()
    if not torch.allclose(nemo_greedy, ort_greedy, rtol=tol, atol=100*tol):
        print(f"Argmax validation failed")


def main():
    #nemo_model = MyEncDecCTCModelBPE.restore_from('/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo')
    nemo_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from('/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo')
    #nemo_model.encoder.export_cache_support = True
    nemo_model = nemo_model.encoder
    nemo_model.export_cache_support = True
    autocast = nullcontext
    with autocast(), torch.no_grad(), torch.inference_mode():
        nemo_model.to(device='cuda').freeze()
        nemo_model.eval()
        input_example = nemo_model.input_example(max_batch=32)

        _, desc = nemo_model.export(
            'tmp.onnx',
            input_example=input_example,
            check_trace=False,
            onnx_opset_version=None,
            verbose=False,
        )

        nemo_outs = nemo_model.forward(
            audio_signal=input_example[0],
            length=input_example[1],
            cache_last_channel=input_example[2],
            cache_last_time=input_example[3],
            #drop_extra_pre_encoded=input_example[4],
        )

    sess = onnxruntime.InferenceSession('tmp.onnx', providers=['CUDAExecutionProvider'])
    for o in sess.get_outputs():
        print(o.name)
    for o in sess.get_inputs():
        print(o.name)

    ort_inputs = {}
    ort_inputs[sess.get_inputs()[0].name] = to_numpy(input_example[0])
    ort_inputs[sess.get_inputs()[1].name] = to_numpy(input_example[1])
    ort_inputs[sess.get_inputs()[2].name] = to_numpy(input_example[2])
    ort_inputs[sess.get_inputs()[3].name] = to_numpy(input_example[3])
    ort_inputs[sess.get_inputs()[4].name] = to_numpy(input_example[4])

    ort_outs = sess.run(None, ort_inputs)
    print(len(ort_outs))
    for o in ort_outs:
        print(o.shape)

    tol = 0.1

    assert len(ort_outs) == len(nemo_outs)
    all_good = True
    for i in range(len(ort_outs)):
        print(f"Testing {sess.get_outputs()[i]}")
        print(nemo_outs[i].size(), ort_outs[i].shape)
        tout = torch.from_numpy(ort_outs[i])
        if not torch.allclose(tout, nemo_outs[i].cpu(), rtol=tol, atol=100*tol):
            print(f"verification vailed for output {sess.get_outputs()[i].name}")
            all_good = False
    if all_good:
        print('Looks Good!')
    else:
        print("Verification failed")


def main_perf():
    asr_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from('/models/sel_ngcinit_nemoasrset3.0_d512_adamwlr2.0_wd0_augx_speunigram1024_streaming_104_12_wm10k_ctc_striding4x_400e_clip1_newcode.nemo').to(torch.device("cuda"))
    with torch.inference_mode():
        with torch.no_grad():
            asr_model.encoder.export_streaming_support = True
            batch_size = 128

            cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
                batch_size=batch_size, device=torch.device("cuda")
            )
            audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"))
            length = torch.full((batch_size,), 57, device=torch.device("cuda"))

            total_audio_len = 0.0
            total_time = 0.0
            for _ in range(384):
                start = time.time()
                (
                    log_probs,
                    encoded_len,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                ) = asr_model.forward_for_export(
                    input=audio_signal,
                    length=length,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                )
                stop = time.time()
                total_time += stop - start
                total_audio_len += (57 / 100) * batch_size
            print(f"Inference took {total_time}s")
            print(f"Total audio {total_audio_len}s")
            print(f"RTFx {total_audio_len / total_time}")


def main_perf_full_context():
    asr_model = nemo_asr.models.ctc_bpe_models.EncDecCTCModelBPE.restore_from('/models/Conformer-CTC-BPE_large_Riva_ASR_set_3.0_ep107.nemo').to(torch.device("cuda"))
    with torch.inference_mode():
        with torch.no_grad():
            batch_size = 128

            audio_signal = torch.randn((batch_size, 80, 57), device=torch.device("cuda"))
            length = torch.full((batch_size,), 57, device=torch.device("cuda"))

            total_audio_len = 0.0
            total_time = 0.0
            for _ in range(384):
                start = time.time()
                log_probs = asr_model.forward_for_export(
                    input=audio_signal,
                    length=length,
                )
                stop = time.time()
                total_time += stop - start
                total_audio_len += (57 / 100) * batch_size
            print(f"Inference took {total_time}s")
            print(f"Total audio {total_audio_len}s")
            print(f"RTFx {total_audio_len / total_time}")

if __name__ == "__main__":
    #main_perf()
    main_perf_full_context()
