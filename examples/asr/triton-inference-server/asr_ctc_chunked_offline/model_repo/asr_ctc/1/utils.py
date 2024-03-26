from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR, AudioFeatureIterator
import numpy as np

class FrameBatchASRWrapper(FrameBatchASR):
    def read_audio_samples(self, samples, delay, model_stride_in_secs):
        samples = np.pad(samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = AudioFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.asr_model.device)
        self.set_frame_reader(frame_reader)

    def predict(self, samples, delay, tokens_per_chunk, model_stride_in_secs):
        self.reset()
        self.read_audio_samples(samples, delay, model_stride_in_secs)
        hyp = self.transcribe(tokens_per_chunk, delay)
        return hyp

    def get_batch_preds(self, batch_samples, delay, tokens_per_chunk, model_stride_in_secs):
        hyps = []
        for samples in batch_samples:
            hyp = self.predict(samples, delay, tokens_per_chunk, model_stride_in_secs)
            hyps.append(hyp)
        return hyps
            

class HFChunkedASR:
    def __init__(self, asr_model, chunk_len_in_secs=30, overlapping=0.5):
        self.asr_model = asr_model
        self.chunk_len_in_secs = chunk_len_in_secs
        self.overlapping = overlapping
        self.chunk_len = int(self.asr_model._cfg.sample_rate * self.chunk_len_in_secs)
        self.overlap_len = int(self.asr_model._cfg.sample_rate * self.overlapping)
    
    def get_batch_preds(self, batch_samples):
        pass