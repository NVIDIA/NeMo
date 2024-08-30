import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import whisper

import torch
import torch.nn.functional as F

from jiwer import wer
from tqdm import tqdm, trange
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.audio import normalize_volume
from torchaudio.transforms import Resample

from scripts.voicebox_edit.speaker_verification.models.ecapa_tdnn import ECAPA_TDNN_SMALL
from scripts.voicebox_edit.data_gen import DataGen

class Eval:
    def __init__(self, wavlm_ckpt="nemo_experiments/checkpoints/wavlm_large_finetune.pth"):
        self.wavlm_ckpt = wavlm_ckpt

    @property
    def voice_encoder(self):
        if not hasattr(self, "_voice_encoder"):
            self._voice_encoder = VoiceEncoder()
        return self._voice_encoder

    @property
    def whisper_model(self):
        if not hasattr(self, "_whisper_model"):
            # self._whisper_model = whisper.load_model("small.en")
            self._whisper_model = whisper.load_model("medium.en")
            # self._whisper_model = whisper.load_model("large-v2")
        return self._whisper_model

    @property
    def wavlm_model(self):
        if not hasattr(self, "_wavlm_model"):
            self._wavlm_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
            if self.wavlm_ckpt is not None:
                state_dict = torch.load(self.wavlm_ckpt, map_location='cpu')
                self._wavlm_model.load_state_dict(state_dict['model'], strict=False)
            self._wavlm_model = self._wavlm_model.to(self.device)
        return self._wavlm_model

    @property
    def device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod    
    def normalize(sentence):
        """Simple text normalization"""
        sentence = sentence.lower()
        
        # remove all punctuation except words and space
        sentence = re.sub(r'[^\w\s]','', sentence)

        sentence = sentence.strip()
        return sentence

    @staticmethod
    def normalize_text(text):
        """MFA text normalization"""
        # Define the punctuation to strip, excluding brackets that should be preserved
        # punctuation_to_strip = r"[、।，@”,:;¿?¡!\\&%#*~，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=]+"
        punctuation_to_strip = r"[、。।，@<>”(),.:;¿?¡!\&%#*~【】，…‥「」『』\"\'_〝〟″⟨⟩♪・‹›«»～′$+=]+"
        
        # Define brackets that should be preserved if they enclose the whole word
        brackets_to_preserve = r"(\[\w+\])|(\{\w+\})|(<\w+>)|(\(\w+\))|(＜\w+＞)"
        
        # Split the text into words using whitespace and additional word break markers, preserving words within brackets
        word_break_markers = r"[\s？!()，,.:;¡¿?“„”&~%#—…‥、。【】$+=〝〟″‹›«»・⟨⟩「」『』”]+"
        words = re.split(word_break_markers, text)
        
        normalized_words = []
        for word in words:
            # Check if the word is enclosed in brackets that should be preserved
            if re.match(brackets_to_preserve, word):
                normalized_words.append(word.lower())
            else:
                # Strip specified punctuation from the beginning and end, then lowercase the word
                word = re.sub(f"^{punctuation_to_strip}|{punctuation_to_strip}$", "", word)
                normalized_words.append(word.lower())
        
        # Rejoin the normalized words into a single string
        return ' '.join(normalized_words)

    @staticmethod
    def create_plot(data, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(data, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(data):
        Eval.create_plot(data.T.cpu(), 'frame', 'freq')

    @staticmethod
    def preprocess_wav(fpath_or_wav: str | np.ndarray, source_sr: int | None = None, db=-30):
        """modified from resemblyzer.process_wav, remove trimming process"""
        # Load the wav from disk if needed
        if isinstance(fpath_or_wav, str):
            wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
        else:
            wav = fpath_or_wav
        
        # Resample the wav
        # if source_sr is not None:
        #     wav = librosa.resample(wav, orig_sr=source_sr, target_sr=16000)
            
        # Apply the preprocessing: normalize volume and shorten long silences 
        wav = normalize_volume(wav, db, increase_only=True)
        
        return wav

    def gen_val_frame_spk_sim(self, data_dir, subset="medium", audio_type="edit", file_type=None):
        """ Calculate frame-level speaker similarity for each audio segment in {data_dir}/{subset}_{file_type}.txt
        Audio segments are decided by real/fake audio boundaries.
        
        Generated: {data_dir}/{subset}_{audio_type}_sim.txt for each {data_dir}/{subset}_{file_type}.txt
        """
        file_type = audio_type if file_type is None else file_type
        with open(f"{data_dir}/{subset}_{file_type}.txt", 'r') as f, open(f"{data_dir}/{subset}_{audio_type}_sim.txt", 'w') as fo:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                fakename, info, label = line.split(' ')
                if line.startswith(f"dev_{audio_type}"):
                    _fname, fname_ = fakename.split('-', 1)
                    filename = f"{_fname.split('_')[-1]}-{fname_}"
                    realname = f"dev_real_{filename}"
                    fake_audio_path = f"{data_dir}/combine/{fakename}.wav"
                    real_audio_path = f"{data_dir}/combine/{realname}.wav"
                    metadata = {
                        "fake_audio_path": fake_audio_path,
                        "real_audio_path": real_audio_path,
                        "info": info,
                    }
                    fake_wav = self.preprocess_wav(fake_audio_path)
                    real_wav = preprocess_wav(real_audio_path)
                    spans = []
                    for span in info.split('/'):
                        span = span.split('-')
                        span, _label = [float(t)*16000 for t in span[:2]], span[2]
                        if _label == "F":
                            spans = span
                            break
                    speaker_emb = self.voice_encoder.embed_utterance(real_wav)
                    _, fake_frame_embs, wav_splits = self.voice_encoder.embed_utterance(fake_wav, return_partials=True, rate=25)
                    frame_sims = fake_frame_embs @ speaker_emb
                    sims = {"bound": [], "in": [], "out": [], "around": []}
                    for i, wav_split in enumerate(wav_splits):
                        if wav_split.start < spans[0]:
                            if wav_split.stop < spans[0]:
                                sims["out"].append(frame_sims[i])
                            else:
                                sims["bound"].append(frame_sims[i])
                                sims["around"].append(frame_sims[i])
                        elif wav_split.start < spans[1]:
                            if wav_split.stop < spans[1]:
                                sims["in"].append(frame_sims[i])
                                sims["around"].append(frame_sims[i])
                            else:
                                sims["bound"].append(frame_sims[i])
                                sims["around"].append(frame_sims[i])
                        else:
                            sims["out"].append(frame_sims[i])
                    for key in sims:
                        sims[key] = np.mean(sims[key]).item()
                                
                    fo.write(' '.join([fakename, info, f"{sims['around']:.3f}", f"{sims['out']:.3f}"]))
                    fo.write('\n')
                    fo.flush()

    def calc_wer(self, audio_path, text):
        whisper_list = []
        gt_list = []
        _whisper_txt = self.whisper_model.transcribe(audio_path)['text']
        whisper_txt = Eval.normalize(_whisper_txt)
        gt_txt = Eval.normalize(text)

        whisper_list.append(whisper_txt)
        gt_list.append(gt_txt)

        whisper_wer = round(wer(gt_list, whisper_list), 5)
        return whisper_wer, _whisper_txt, whisper_txt

    def calc_wers(self, audio_paths, texts, output_file=None):
        whisper_list = []
        gt_list = []
        if output_file is not None:
            fo = open(output_file, 'w')
            fo.write("gt_txt\twhisper.txt\twer\n")
        for audio_path, text in zip(audio_paths, texts):
            whisper_txt = Eval.normalize(self.whisper_model.transcribe(audio_path)['text'])
            gt_txt = Eval.normalize(text)

            whisper_list.append(whisper_txt)
            gt_list.append(gt_txt)

            # print(whisper_txt)
            # print(gt_txt)
            if output_file is not None:
                fo.write(f"{gt_txt}\t{whisper_txt}\t{round(wer([gt_txt], [whisper_txt]), 5)}\n")
                # fo.write(f"{round(wer([gt_txt], [whisper_txt]), 5)}\n")

        whisper_wer = round(wer(gt_list, whisper_list), 5)
        if output_file is not None:
            fo.close()
        return whisper_wer

    def calc_wavlm_sim(self, wav1, wav2):
        wav1, sr1 = sf.read(wav1)
        wav2, sr2 = sf.read(wav2)

        wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
        wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
        resample1 = Resample(orig_freq=sr1, new_freq=16000)
        resample2 = Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        model = self.wavlm_model
        wav1 = wav1.to(self.device)
        wav2 = wav2.to(self.device)

        model.eval()
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)
        return sim.cpu().item()