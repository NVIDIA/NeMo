#%%
import os
import sys
import random
import json
import soundfile as sf
import torch
import torchaudio
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from einops import rearrange
from jiwer import wer

from nemo.collections.tts.models.voicebox import VoiceboxModel
from scripts.voicebox_edit.data_gen import DataGen
from scripts.voicebox_edit.eval import Eval

sys_prompt = """Given a transcript of a speech, your task is to subtly alter its meaning by changing only a single word or phrase. This change should pivot the original intent or information conveyed, such as transforming a word to its antonym, substituting a name or a noun with another, or slightly altering a phrase that shifts the narrative direction. The challenge lies in making this change impactful enough to alter the transcript's overall meaning, while ensuring that the modification is limited to a very small part of the text. The rest of the transcript must remain untouched.

**Objective:** Focus on identifying a pivotal word or phrase whose modification can flip the narrative or significantly change the message, with minimal intervention.

**Constraints:**
- Only one word or phrase may be altered.
- The alteration should substantially change the meaning of the transcript.
- All other words in the transcript must remain exactly as they are.
- The modified word or phrase should constitute a small ratio of the text to ensure the exercise's subtlety.

**Output Requirement:** Provide only the modified transcript. Do not include any explanations or annotations.

**Example:**

- **Original Transcript:** "There's a llama on my lawn, how can I get rid of him?"
- **Modified Transcript:** "There's a lion on my lawn, how can I get rid of him?"

Proceed by applying this instruction to the given transcript, ensuring the modification adheres to the outlined constraints and objectives."""

class DataProcessor:
    def __init__(self, model: VoiceboxModel):
        self.model = model

    def get_riva_demo_data(self, output_dir="nemo_experiments/riva_demo_gen"):
        os.makedirs(output_dir, exist_ok=True)
        audio_names = []
        audio_data = [
            {
                "name": "LINDY_CMU_ANGRY_000407",
                "text": "Mercedes screamed, cried, laughed, and manifested the chaotic abandonment of hysteria.",
                "from": "of hysteria",
                "time": 6.64,
            },
            {
                "name": "LINDY_CMU_CALM_000407",
                "text": "Mercedes screamed, cried, laughed, and manifested the chaotic abandonment of hysteria.",
                "from": "of hysteria",
                "time": 7.3,
            },
            {
                "name": "LINDY_CMU_DISGUSTED_000023",
                "text": "A combination of Canadian capital quickly organized and petitioned for the same privileges. ",
                "from": "for the same privileges",
                "time": 6.18,
            },
            {
                "name": "LINDY_CMU_HAPPY_000472",
                "text": "He is too keenly intelligent, too sharply sensitive, successfully to endure.",
                "from": "to endure",
                "time": 5.15,
            },
            {
                "name": "LINDY_CMU_SAD_000023",
                "text": "A combination of Canadian capital quickly organized and petitioned for the same privileges. ",
                "from": "for the same privileges",
                "time": 5.77,
            },
            {
                "name": "LINDY_pa",
                "text": "Other sources put the numbers of speakers at one hundred eighty thousand, two hundred twenty thousand, and two hundred fifty thousand, whereas Yugoslav sources vary, some putting the estimated number of Macedonians in Greek Macedonia at one hundred fifty thousand to two hundred thousand and others at three hundred thousand.",
                "from": "at three hundred thousand",
                "time": 17.12,
            },
            {
                "name": "RODNEY_CMU_ANGRY_000407",
                "text": "Mercedes screamed, cried, laughed, and manifested the chaotic abandonment of hysteria.",
                "from": "of hysteria",
                "time": 6.87,
            },
            {
                "name": "RODNEY_CMU_CALM_000407",
                "text": "Mercedes screamed, cried, laughed, and manifested the chaotic abandonment of hysteria.",
                "from": "of hysteria",
                "time": 6.52,
            },
            {
                "name": "RODNEY_CMU_DISGUSTED_000023",
                "text": "A combination of Canadian capital quickly organized and petitioned for the same privileges. ",
                "from": "for the same privileges",
                "time": 7.67,
            },
            {
                "name": "RODNEY_CMU_HAPPY_000023",
                "text": "A combination of Canadian capital quickly organized and petitioned for the same privileges. ",
                "from": "for the same privileges",
                "time": 4.27,
            },
            {
                "name": "RODNEY_CMU_SAD_000407",
                "text": "Mercedes screamed, cried, laughed, and manifested the chaotic abandonment of hysteria.",
                "from": "of hysteria",
                "time": 7.07,
            },
            # {
            #     "name": "RODNEY_pa",
            #     "text": "During two thousand one, according to several international sources, twenty eight to thirty thousand Pakistani nationals, fourteen to fifteen thousand Afghan Taliban, and two to three thousand al-Qaeda militants were fighting against anti-Taliban forces in Afghanistan as a roughly fourty five thousand strong military force.",
            #     "from": "as a roughly fourty five thousand strong military force",
            #     "time": 16.05,
            # },
            {
                "name": "voice_prompt",
                "text": "What are you talking about, man? said one of the bystanders. I have got them. I have got them. And they are not worth three reels.",
                "from": "worth three reels",
            },
        ]
        for i, _ in enumerate(audio_data):
            if audio_data[i]["name"][:5] == "LINDY":
                spk = "Lindy"
            elif audio_data[i]["name"][:6] == "RODNEY":
                spk = "Rodney"
            else:
                spk = "Other"

            name = audio_data[i]["name"]
            if name == "voice_prompt":
                audio_data[i].update({
                    "audio_path": f"nemo_experiments/RIVA/{spk}/{name}.wav",
                })
            else:
                audio_data[i].update({
                    "audio_path": f"nemo_experiments/RIVA/{spk}/{name}.wav",
                    "textgrid_path": f"nemo_experiments/RIVA_MFA/{spk}/{name}.TextGrid",
                })

        text_data = [
            "In different countries, there are different requirements for an individual to legally practice neurosurgery, and there are varying methods through which they must be educated.",
            "I help design video games for a living. More on the concept side, less on the actual building! Do you play?",
            "There are over five hundred federally recognized tribes within the U.S., about half of which are associated with Indian reservations.",
            "I shop online way too much. I tend to visit too many retailers and searching for products I might want or need and it shows me everything I want to know.",
            "Single-payer healthcare is a healthcare system financed by taxes that covers the costs of essential healthcare for all residents, with costs covered by a single public system.",
            "Absolutely, I love prime and being able to buy whatever I want from the largest internet retailer in the world.",
        ]

        data = []
        for a_data in audio_data:
            for t_data in text_data:
                data.append({
                    "audio_path": a_data["audio_path"],
                    "text": a_data["text"],
                    "from": a_data["from"],
                    "to": t_data,
                    "out_ori_path": f"{output_dir}/{a_data['name']}_ori.wav",
                    "out_gen_path": f"{output_dir}/{a_data['name']}-{t_data}_gen.wav",
                    "out_tts_path": f"{output_dir}/{a_data['name']}-{t_data}_tts.wav",
                })
                if "textgrid_path" in a_data:
                    data[-1].update({
                        "textgrid_path": a_data["textgrid_path"],
                    })
                if "time" in a_data:
                    data[-1].update({
                        "time": a_data["time"],
                    })

        return data

    def get_RealEdit_data(self, realedit_dir="nemo_experiments/RealEdit", filepath="nemo_experiments/RealEdit/RealEdit.tsv", output_dir="nemo_experiments/RealEdit/gen"):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}_ori", exist_ok=True)
        df = pd.read_csv(filepath, sep='\t')

        data = []
        for row in df.itertuples():
            # multi-span checking
            if '|' in row.orig_transcript:
                assert '|' in row.new_transcript
                assert '|' in row.orig_masked_span
                assert '|' in row.new_masked_span
                assert '|' in row.type

            data.append({
                "audio_path": f"{realedit_dir}/Original/{row.wav_fn}",
                "VC_audio_path": f"{realedit_dir}/e6_2_20_400_beam1_topp0.8_temp1_topk0_selected/{row.wav_fn[:-4]}_new_seed1.wav",
                "text": row.orig_transcript,
                "target": row.new_transcript,
                "edit_pos": row.orig_masked_span,
                "target_pos": row.new_masked_span,
                "edit_type": row.type,
                "out_ori_path": f"{output_dir}_ori/{row.wav_fn}",
                "out_gen_path": f"{output_dir}/{row.wav_fn}",
                "out_tts_path": f"{output_dir}/tts_{row.wav_fn}",
            })
            
            textgrid_path = f"{realedit_dir}/Original_MFA/{row.wav_fn[:-4]}.TextGrid"
            if os.path.exists(textgrid_path):
                data[-1].update({
                    "textgrid_path": textgrid_path,
                })
            else:
                print(row.wav_fn)

        return data

class Inference:
    def __init__(self, model: VoiceboxModel, sample_std=0.9):
        self.model = model
        self.sample_std = sample_std
        self.mfa_en_dict = {}
        with open("/root/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict", 'r') as f:
            for line in tqdm(f):
                wrd, *_, phns = line.strip().split('\t')
                if wrd not in self.mfa_en_dict:
                    self.mfa_en_dict[wrd] = phns

    def internal_demo(self, data, ztts=False):
        # shape: (1, L), (1,), scalar
        wav_input, fs = torchaudio.load(data["audio_path"])
        transform5 = torchaudio.transforms.Resample(fs, self.model.voicebox.audio_enc_dec.sampling_rate)
        audio_data = transform5(wav_input).squeeze(0).numpy()
        audio = torch.tensor(audio_data, dtype=torch.float, device=self.model.device).unsqueeze(0)
        audio_len = torch.tensor(audio.shape[1], device=self.model.device).unsqueeze(0)
        
        edit_pred = self.model.forward(
            audio=audio,
            audio_lens=audio_len,
            texts=[data["text"],],
            textgrids=None if "textgrid_path" not in data else [data["textgrid_path"],],
            edit_from=[data["from"],],
            edit_to=[data["to"],],
            steps=64,
            cond_scale=1.0,
            sample_std=self.sample_std,
            dp_scale=1.2,
            ztts=ztts,
            edit_alignments=None if "edit_alignment" not in data else [data["edit_alignment"]],
            mfa_en_dict=self.mfa_en_dict,
        )

        sf.write(data["out_ori_path"], audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        edit_audio = edit_pred["edit_audio"][0].cpu().numpy()
        sf.write(data["out_gen_path"], edit_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
        if ztts:
            ztts_audio = edit_pred["ztts_audio"][0].cpu().numpy()
            sf.write(data["out_tts_path"], ztts_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        return edit_pred["ori_mel"], edit_pred["edit_mel"]

    def riva_demo(self, data, ztts=False):
        # shape: (1, L), (1,), scalar
        wav_input, fs = torchaudio.load(data["audio_path"])
        transform5 = torchaudio.transforms.Resample(fs, self.model.voicebox.audio_enc_dec.sampling_rate)
        audio_data = transform5(wav_input).squeeze(0).numpy()
        audio = torch.tensor(audio_data, dtype=torch.float, device=self.model.device).unsqueeze(0)
        audio_len = torch.tensor(audio.shape[1], device=self.model.device).unsqueeze(0)
        
        edit_pred = self.model.forward(
            audio=audio,
            audio_lens=audio_len,
            texts=[data["text"],],
            textgrids=None if "textgrid_path" not in data else [data["textgrid_path"],],
            edit_from=[data["from"],],
            edit_to=[data["to"],],
            steps=64,
            cond_scale=1.0,
            sample_std=self.sample_std,
            dp_scale=1.2,
            ztts=ztts,
            edit_alignments=None if "edit_alignment" not in data else [data["edit_alignment"]],
            mfa_en_dict=self.mfa_en_dict,
        )

        sf.write(data["out_ori_path"], audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        edit_audio = edit_pred["edit_audio"][0].cpu().numpy()
        if "time" in data:
            # to cut off ref audio
            edit_audio = edit_audio[int(data["time"] * _sr):]
        sf.write(data["out_gen_path"], edit_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        if ztts:
            ztts_audio = edit_pred["ztts_audio"][0].cpu().numpy()
            sf.write(data["out_tts_path"], ztts_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        return edit_pred["ori_mel"], edit_pred["edit_mel"]

    def RealEdit(self, data, ztts=False, tag="", margin=3, redit=True, tune_vol=False):
        def get_span(span):
            span = span.split(',')
            if len(span) == 1:
                span = span * 2
            return [int(sp) for sp in span]

        def apply_span(trans, span):
            trans = trans.split(' ')
            span = get_span(span)
            return ' '.join(trans[span[0]: span[1]+1])

        def span_shift(from_spans, to_spans, edit_types, orig_len):
            """
            Return:
                thres: id <= thres: keep
            """
            ids = list(range(orig_len))
            outputs = []
            for from_span, to_span, edit_type in zip(from_spans, to_spans, edit_types):
                assert edit_type in ["insertion", "deletion", "substitution"]
                from_span = [ids.index(from_span[0]), ids.index(from_span[1])]
                outputs.append(from_span)

                if edit_type == "insertion":
                    ids = ids[:from_span[0]+1] + [-1]*(to_span[1]-to_span[0]+1) + ids[from_span[1]:]
                elif edit_type == "deletion":
                    ids = ids[:from_span[0]] + ids[from_span[1]+1:]
                elif edit_type == "substitution":
                    ids = ids[:from_span[0]] + [-1]*(to_span[1]-to_span[0]+1) + ids[from_span[1]+1:]

            return outputs

        # shape: (1, L), (1,), scalar
        wav_input, fs = torchaudio.load(data["audio_path"])
        transform5 = torchaudio.transforms.Resample(fs, self.model.voicebox.audio_enc_dec.sampling_rate)
        audio_data = transform5(wav_input).squeeze(0).numpy()
        audio = torch.tensor(audio_data, dtype=torch.float, device=self.model.device).unsqueeze(0)
        audio_len = torch.tensor(audio.shape[1], device=self.model.device).unsqueeze(0)
        if tune_vol:
            int16_max = ((2 ** 15) - 1)
            rms = np.sqrt(np.mean((audio_data * int16_max) ** 2))

        # multi-span
        text_list = data["text"].split('|')
        edit_pos_list = data["edit_pos"].split('|')
        edit_type_list = data["edit_type"].split('|')
        target_list = data["target"].split('|')
        tgt_pos_list = data["target_pos"].split('|')
        from_list = [apply_span(text_list[0], edit_pos) for edit_pos in edit_pos_list]
        to_list = [apply_span(target, tgt_pos) for target, tgt_pos in zip(target_list, tgt_pos_list)]

        from_spans = [get_span(edit_pos) for edit_pos in edit_pos_list]
        to_spans = [get_span(tgt_pos) for tgt_pos in tgt_pos_list]
        shifted_spans = span_shift(from_spans, to_spans, edit_type_list, len(text_list[0].split(' ')))

        edit_pred = None
        for i, (_text, _from, _to, _e_pos, _e_type) in enumerate(zip(text_list, from_list, to_list, edit_pos_list, edit_type_list)):
            _text = DataGen.normalize_text(_text).strip()
            _from = DataGen.normalize_text(_from).strip()
            _to = DataGen.normalize_text(_to).strip()
            _e_pos = f"{shifted_spans[i][0]},{shifted_spans[i][1]}"
            if edit_pred is not None:
                audio = edit_pred["redit_audio"]
                audio_len = edit_pred["new_audio_lens"]
                if tune_vol:
                    _rms = torch.sqrt(torch.mean((edit_pred["redit_audio"][0, edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] * int16_max) ** 2))
                    with torch.no_grad():
                        audio = audio.clone()
                        audio[0, edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] = edit_pred["redit_audio"][0, edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] * rms / _rms

            edit_pred = self.model.forward(
                audio=audio,
                audio_lens=audio_len,
                texts=[_text,],
                textgrids=None if (edit_pred is not None or "textgrid_path" not in data) else [data["textgrid_path"],],
                edit_from=[_from,],
                edit_to=[_to,],
                steps=64,
                cond_scale=1.0,
                sample_std=self.sample_std,
                dp_scale=1.2,
                ztts=ztts,
                edit_alignments=None,
                edit_positions=[_e_pos,],
                edit_types=[_e_type,],
                mfa_en_dict=self.mfa_en_dict,
                margin=margin,
            )

        sf.write(data["out_ori_path"], audio[0].cpu().numpy(), samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        data["_out_gen_path"] = data["out_gen_path"][:-4]+f"{tag}.wav"
        edit_audio = edit_pred["edit_audio"][0].cpu().numpy()
        if tune_vol:
            _rms = np.sqrt(np.mean((edit_audio[edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] * int16_max) ** 2))
            edit_audio[edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] = edit_audio[edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] * rms / _rms
        sf.write(data["_out_gen_path"], edit_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        if ztts:
            data["_out_tts_path"] = data["out_tts_path"][:-4]+f"{tag}.wav"
            ztts_audio = edit_pred["ztts_audio"][0].cpu().numpy()
            sf.write(data["_out_tts_path"], ztts_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        if redit:
            data["_out_redit_path"] = data["out_gen_path"][:-4]+f".redit{tag}.wav"
            redit_audio = edit_pred["redit_audio"][0].cpu().numpy()
            if tune_vol:
                _rms = np.sqrt(np.mean((redit_audio[edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] * int16_max) ** 2))
                redit_audio[edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] = redit_audio[edit_pred["new_cond_st_idx"].item():edit_pred["new_cond_ed_idx"].item()] * rms / _rms
            sf.write(data["_out_redit_path"], redit_audio, samplerate=self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
            data["_out_edit_path"] = data["_out_gen_path"]
            data["_out_gen_path"] = data["_out_redit_path"]

        return data, edit_pred

class MainExc:
    def __init__(self, vb_ckpt_path=None, dp_ckpt_path=None, gen_data_dir="data/gen_dataset", sample_std=0.9):
        self.vb_ckpt_path = vb_ckpt_path
        self.dp_ckpt_path = dp_ckpt_path

        self.gen_data_dir = gen_data_dir
        self.sample_std = sample_std

    def load_model(self,):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = VoiceboxModel.load_from_checkpoint(self.vb_ckpt_path, map_location=device)
        model = VoiceboxModel.load_from_checkpoint(self.vb_ckpt_path, map_location=device, strict=False)

        # dp_model = VoiceboxModel.load_from_checkpoint(self.dp_ckpt_path, map_location=device)
        dp_model = VoiceboxModel.load_from_checkpoint(self.dp_ckpt_path, map_location=device, strict=False)

        del model.duration_predictor, model.cfm_wrapper.duration_predictor
        model.duration_predictor = dp_model.duration_predictor
        model.cfm_wrapper.duration_predictor = dp_model.duration_predictor
        del dp_model

        torch.cuda.empty_cache()

        model.cap_vocode = True
        return model

    def prepare_val_dl(self, ds_name="libriheavy", corpus_dir="data/download/LibriLight/", manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_dev.jsonl.gz",
                       old_prefix="download/librilight", min_duration=-1, max_duration=float("inf"), load_audio=True, filter_ids=None, shuffle=False, batch_duration=100):
        # load from val set
        self.model.cfg.ds_name = ds_name
        self.model.cfg.corpus_dir = corpus_dir
        self.model.cfg.validation_ds.manifest_filepath = manifest_filepath
        self.model.cfg.validation_ds.lhotse.cuts_path = self.model.cfg.validation_ds.manifest_filepath
        with open_dict(self.model.cfg.validation_ds):
            self.model.cfg.validation_ds.min_duration = min_duration
            self.model.cfg.validation_ds.max_duration = max_duration
            self.model.cfg.validation_ds.ds_kwargs.load_audio = load_audio
            self.model.cfg.validation_ds.filter_ids = filter_ids
            self.model.cfg.validation_ds.num_workers = 8
            self.model.cfg.validation_ds.shuffle = shuffle
            self.model.cfg.validation_ds.batch_duration = batch_duration
        with open_dict(self.model.cfg):
            self.model.cfg["old_prefix"] = old_prefix
        self.model.setup_validation_data(self.model.cfg.validation_ds)

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.load_model()
        return self._model

    @property
    def dataprocessor(self):
        if not hasattr(self, "_dataprocessor"):
            self._dataprocessor = DataProcessor(model=self.model)
        return self._dataprocessor

    @property
    def datagen(self):
        if not hasattr(self, "_datagen"):
            self._datagen = DataGen(model=self.model, sample_std=self.sample_std)
        return self._datagen

    @property
    def infer(self):
        if not hasattr(self, "_infer"):
            self._infer = Inference(model=self.model, sample_std=self.sample_std)
        return self._infer
    
    @property
    def eval(self):
        if not hasattr(self, "_eval"):
            self._eval = Eval()
        return self._eval

    def prepare_gigaspeech_val_dl(self, manifest_filepath="data/parsed/GigaSpeech/gigaspeech_cuts_DEV.speech.jsonl.gz"):
        ds_name="gigaspeech"
        corpus_dir="data/download/GigaSpeech"
        self.prepare_val_dl(ds_name="gigaspeech",
                            corpus_dir="data/download/GigaSpeech",
                            manifest_filepath=manifest_filepath,
                            old_prefix="/home/sungfengh/.cache/huggingface/datasets")

    def gen_val_v1(self,):
        # self.prepare_val_dl()
        self.prepare_gigaspeech_val_dl()
        self.datagen.gen_v1_dataset_from_val_set("nemo_experiments/gen_dataset/dev-v1")

    def gen_v3_transcript_json(self):
        """generate json for LLM transcript editing"""
        self.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_small.jsonl.gz", min_duration=4, max_duration=10, load_audio=False)
        self.datagen.gen_edit_transcript_json(f"{self.gen_data_dir}/small_prompt.json")

        self.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_medium.jsonl.gz", min_duration=6, max_duration=8, load_audio=False)
        self.datagen.gen_edit_transcript_json(f"{self.gen_data_dir}/medium_prompt.json")

    def gen_v3(self, split_id=None, out_dict=None, tag="",
                gpt_file="nemo_experiments/data_1a_medium.json", out_dict_file="nemo_experiments/data_parsed_1a_medium.json"):
        """ generate SINE dataset """
        if not out_dict:
            if not os.path.exists(out_dict_file):
                out_dict = self.datagen.load_gpt_json(gpt_file, out_dict_file)
            else:
                out_dict = json.load(open(out_dict_file, 'r'))

        # select a split to generate
        if split_id is None:
            split_id = int(sys.argv[1])
        filter_ids = sorted(out_dict.keys())[split_id*3000: (split_id+1)*3000]

        self.prepare_val_dl(manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_medium.jsonl.gz",
                            min_duration=6, max_duration=8,
                            filter_ids=filter_ids)

        self.datagen.gen_v3_dataset_from_val_set(out_dict, f"{self.gen_data_dir}/medium-v3{tag}/split-{split_id}")

        self.eval.gen_val_frame_spk_sim(data_dir=f"{self.gen_data_dir}/medium-v3{tag}/split-{split_id}", subset="medium", audio_type="edit")

    def calc_dac_stats(self, ds_name="gigaspeech", corpus_dir="data/download/GigaSpeech", manifest_filepath="data/parsed/GigaSpeech/gigaspeech_cuts_DEV.speech.jsonl.gz", shuffle=False):
        self.prepare_val_dl(ds_name=ds_name, corpus_dir=corpus_dir, manifest_filepath=manifest_filepath, old_prefix="/home/sungfengh/.cache/huggingface/datasets", shuffle=shuffle)
        self.datagen.get_dac_statistics()

    def riva_demo(self, output_dir="nemo_experiments/riva_demo_gen"):
        datas = self.dataprocessor.get_riva_demo_data(output_dir)
        for data in datas:
            ori_mel, edit_mel = self.infer.riva_demo(data)

    def gen_RealEdit(self, realedit_dir="nemo_experiments/RealEdit", output_dir="nemo_experiments/RealEdit/gen", regen=False, tune_vol=False):
        datas = self.dataprocessor.get_RealEdit_data(filepath=f"{realedit_dir}/RealEdit.tsv", output_dir=output_dir)
        rounds = 5
        text_list = []
        audio_file_list = []
        gen_audio_file_list = []
        gen_sims = []
        gt_txt_list = []
        whisper_list = []
        for i in range(rounds):
            whisper_list.append([])

        os.makedirs(f"{output_dir}/metadata", exist_ok=True)
        loaded = False if regen else os.path.exists(f"{output_dir}/metadata/metadata.tsv")
        if loaded:
            df = pd.read_csv(f"{output_dir}/metadata/metadata.tsv", sep='\t')
            with open(f"{output_dir}/metadata/metadata.tsv", 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    line = line.strip('\n').split('\t')
                    for j in range(rounds):
                        gen_audio_file_list.append(line[0])
            f = open(f"{output_dir}/metadata/metadata.tsv", 'a')
            row_id = 0
        else:
            f = open(f"{output_dir}/metadata/metadata.tsv", 'w')
            for i in range(rounds):
                f.write(f"wav_fn_{i}\twhisper_txt_{i}\tnorm_whisper_txt_{i}\twer_{i}\tsim_{i}\t")
            f.write(f"gt_txt\tnorm_gt_txt\n")

        for i, data in tqdm(enumerate(datas)):
            target_list = data["target"].split('|')
            text_list.append(target_list[-1])
            gt_txt = DataGen.normalize(target_list[-1])
            gt_txt_list.append(gt_txt)

            audio_file_list.append(data["out_ori_path"])

            gen_audio_path = None

            if loaded and i < len(gen_audio_file_list):
                gen_audio_path = gen_audio_file_list[i]
                gen_sim = self.eval.calc_wavlm_sim(data["out_ori_path"], gen_audio_path)
                gen_sims.append(gen_sim)
            else:
                _gen_sims = []
                _gen_audio_paths = []
                for j in range(rounds):
                    try:
                        data, edit_pred = self.infer.RealEdit(data, tag=f".round-{j}", tune_vol=tune_vol)
                    except:
                        raise
                        continue

                    gen_sim = self.eval.calc_wavlm_sim(data["out_ori_path"], data["_out_gen_path"])
                    _gen_sims.append(gen_sim)
                    _gen_audio_paths.append(data["_out_gen_path"])

                    gen_wer, whisper_txt, norm_whisper_txt = self.eval.calc_wer(audio_path=data["_out_gen_path"], text=target_list[-1])
                    whisper_list[j].append(whisper_txt)
                    f.write(f"{data['_out_gen_path']}\t{whisper_txt}\t{norm_whisper_txt}\t{gen_wer:.5f}\t{gen_sim:.5f}\t")

                max_id = _gen_sims.index(max(_gen_sims))
                _gen_sims.pop(max_id)
                _gen_audio_paths.pop(max_id)
                min_id = _gen_sims.index(min(_gen_sims))
                _gen_sims.pop(min_id)
                _gen_audio_paths.pop(min_id)
                _id = random.randrange(len(_gen_sims))
                gen_sim = _gen_sims[_id]
                gen_audio_path = _gen_audio_paths[_id]

                gen_sims.append(gen_sim)
                gen_audio_file_list.append(gen_audio_path)

                f.write(f"{target_list[-1]}\t{gt_txt}\n")
                f.flush()


        avg_wer = self.eval.calc_wers(audio_paths=gen_audio_file_list, texts=text_list, output_file=f"{output_dir}.tsv")
        avg_sim = sum(gen_sims) / len(gen_sims)

        print(f"WER: {avg_wer}")
        print(f"SIM: {avg_sim}")

        with open(f"{output_dir}/metadata/score.tsv", 'w') as fs:
            fs.write(f"wer\tsim\n")
            fs.write(f"{avg_wer}\t{avg_sim:.5f}")
        f.close()


if __name__ == "__main__":
    kwargs = {
        "unet": {
            "main_exc": {
                "vb_ckpt_path": "nemo_experiments/checkpoints/a100-GS_XL-DAC-pymha-unet-warmup/checkpoints/vb-val_loss/vb=0.2913-epoch=167-step=500000-last.ckpt",
                "dp_ckpt_path": "nemo_experiments/checkpoints/dp_no_sil_spn=1.4410-epoch=8.ckpt",
            },
            "riva_demo": {"output_dir": "nemo_experiments/riva_demo_gen_gs_unet"},
            "RealEdit": {"realedit_dir": "nemo_experiments/RealEdit", "output_dir": "nemo_experiments/RealEdit/gen_gs_unet"},
        },
        "unet_noCE": {
            "main_exc": {
                "vb_ckpt_path": "nemo_experiments/checkpoints/a100-GS_XL-DAC_noCE-pymha-unet-warmup/checkpoints/vb-val_loss/vb=0.2933-epoch=167-step=500000-last.ckpt",
                "dp_ckpt_path": "nemo_experiments/checkpoints/dp_no_sil_spn=1.4410-epoch=8.ckpt",
            },
            "riva_demo": {"output_dir": "nemo_experiments/riva_demo_gen_gs_unet_noCE"},
            "RealEdit": {"realedit_dir": "nemo_experiments/RealEdit", "output_dir": "nemo_experiments/RealEdit/gen_gs_unet_noCE"},
        },
        "unet-mel": {
            "main_exc": {
                "vb_ckpt_path": "nemo_experiments/checkpoints/a100-GS_XL-mel-pymha-unet-warmup/checkpoints/vb-val_loss/vb=0.2564-epoch=153-step=462000.ckpt",
                "dp_ckpt_path": "nemo_experiments/checkpoints/1b_oci_voicebox--val_loss_total=3.2725-epoch=61.ckpt",
            },
            "riva_demo": {"output_dir": "nemo_experiments/riva_demo_gen_gs_unet-mel"},
            "RealEdit": {"realedit_dir": "nemo_experiments/RealEdit", "output_dir": "nemo_experiments/RealEdit/gen_gs_unet-mel"},
        },
        "unet-lh-mel": {
            "main_exc": {
                "vb_ckpt_path": "nemo_experiments/checkpoints/a100-LH_M-mel-pymha-unet-warmup/checkpoints/vb-val_loss/vb=0.2742-epoch=167-step=500000-last.ckpt",
                "dp_ckpt_path": "nemo_experiments/checkpoints/1b_oci_voicebox--val_loss_total=3.2725-epoch=61.ckpt",
            },
            "riva_demo": {"output_dir": "nemo_experiments/riva_demo_gen_gs_unet-lh-mel"},
            "RealEdit": {"realedit_dir": "nemo_experiments/RealEdit", "output_dir": "nemo_experiments/RealEdit/gen_gs_unet-lh-mel"},
        },
        "unet_postq": {
            "main_exc": {
                "vb_ckpt_path": "nemo_experiments/checkpoints/a100-GS_XL-DAC_postq-pymha-unet-warmup/checkpoints/vb-val_loss/vb=0.4783-epoch=167-step=500000-last.ckpt",
                "dp_ckpt_path": "nemo_experiments/checkpoints/dp_no_sil_spn=1.4410-epoch=8.ckpt",
            },
            "riva_demo": {"output_dir": "nemo_experiments/riva_demo_gen_gs_unet_postq"},
            "RealEdit": {"realedit_dir": "nemo_experiments/RealEdit", "output_dir": "nemo_experiments/RealEdit/gen_gs_unet_postq"},
        },
    }
    # exp = "unet-mel"
    # main_exc = MainExc(**(kwargs[exp]["main_exc"]), gen_data_dir="nemo_experiments/gen_dataset", sample_std=0.95)
    # main_exc.gen_val_v1()
    # main_exc.riva_demo(**(kwargs[exp]["riva_demo"]))
    # main_exc.gen_v3(split_id=None, tag="-unet-500k")
    # for exp in ["unet-mel"]:
    # for exp in [ "unet-mel", "unet-lh-mel"]:
    for exp in ["unet", "unet_noCE", "unet_postq", "unet-lh-mel", "unet-mel"]:
        main_exc = MainExc(**(kwargs[exp]["main_exc"]), gen_data_dir="nemo_experiments/gen_dataset", sample_std=0.95)
        main_exc.gen_RealEdit(**(kwargs[exp]["RealEdit"]), regen=True, tune_vol=False)
        del main_exc

    # main_exc.calc_dac_stats(ds_name="gigaspeech", corpus_dir="data/download/GigaSpeech", manifest_filepath="data/parsed/GigaSpeech/gigaspeech_cuts_XL.speech.jsonl.gz", shuffle=True)
    