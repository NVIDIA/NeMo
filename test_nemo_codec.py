import torch
import torchaudio

from nemo.collections.tts.models import AudioCodecModel

checkpoint_path = '/Data/Checkpoints/rlang_codec/SpeechCodec.nemo'
device = 'cuda'
audio_codec = AudioCodecModel.restore_from(checkpoint_path).to(device).eval()

input_file_path = "/Data/LibriTTS/train-clean-360/986/129388/986_129388_000002_000007.wav"
audio_codec = audio_codec.to(device)
# signal = AudioSignal(input_file_path)
wav, sr = torchaudio.load(input_file_path)
if sr != audio_codec.sample_rate:
    wav = torchaudio.transforms.Resample(sr, audio_codec.sample_rate)(wav)


print(wav.shape)
print(len(wav[-1]))
audio_len = torch.Tensor([len(wav[-1])]).cuda()
print(audio_len)
wav = wav.cuda()

# Encode audio into token indices [batch_size, num_codebooks, time]
tokens, token_lens = audio_codec.encode(audio=wav, audio_len=audio_len)
print('Tokens shape:', tokens.shape)
print('Token_lens:', token_lens)
# Decode audio from token indices.
audio_pred, audio_pred_lens = audio_codec.decode(tokens=tokens, tokens_len=token_lens)

print('Audio pred shape:', audio_pred.shape)

torchaudio.save("/Data/test_newcode.wav", audio_pred.cpu(), audio_codec.sample_rate)

print("Saved", "/Data/test_newcode.wav")