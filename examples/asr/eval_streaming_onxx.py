import nemo.collections.asr as nemo_asr
import onnxruntime
import torch

def to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


asr_model_path = '/drive3/checkpoints/streaming/causal_model_new_chunked.nemo'
onnx_model_path = '/drive3/checkpoints/streaming/causal_model_new_chunked2.onnx'

asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=asr_model_path)
onnx_model = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

asr_model.encoder.export_cache_support = True
processed_signal, processed_signal_length, cache_last_channel, cache_last_time = asr_model.encoder.input_example(max_batch=1, max_dim=4096)

encoder_output_pt = asr_model.encoder.forward(
    audio_signal=processed_signal,
    length=processed_signal_length,
    cache_last_channel=cache_last_channel,
    cache_last_time=cache_last_time,
)

if len(encoder_output_pt) == 2:
    encoded_pt, encoded_len_pt = encoder_output_pt
    cache_last_channel_next_pt = cache_last_time_next_pt = None
else:
    encoded_pt, encoded_len_pt, cache_last_channel_next_pt, cache_last_time_next_pt = encoder_output_pt



ort_inputs = {
    onnx_model.get_inputs()[0].name: to_numpy(processed_signal),
    onnx_model.get_inputs()[1].name: to_numpy(processed_signal_length),
    onnx_model.get_inputs()[2].name: to_numpy(cache_last_channel),
    onnx_model.get_inputs()[3].name: to_numpy(cache_last_time),
}
encoder_output_onnx = onnx_model.run(None, ort_inputs)
for idx, t in enumerate(encoder_output_onnx):
    encoder_output_onnx[idx] = torch.tensor(t).to(processed_signal.device)
if len(encoder_output_onnx) == 2:
    encoded_onnx, encoded_len_onnx = encoder_output_onnx
    cache_last_channel_next_onnx = cache_last_time_next_onnx = None
else:
    encoded_onnx, encoded_len_onnx, cache_last_channel_next_onnx, cache_last_time_next_onnx = encoder_output_onnx

print(cache_last_time_next_pt)
print(cache_last_time_next_onnx)