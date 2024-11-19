import os
import re
import random
import json
import jiwer
import soundfile as sf
import numpy as np

import torch

from einops import rearrange
from tqdm import tqdm

from nemo.collections.tts.models.voicebox import VoiceboxModel, fix_alignment


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

class DataGen:
    def __init__(self, model: VoiceboxModel, sample_std=.95):
        self.model = model
        self.sample_std = sample_std
        self.mfa_en_dict = {}
        with open("/root/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict", 'r') as f:
            for line in tqdm(f):
                wrd, _, _, _, _, phns = line.strip().split('\t')
                if wrd not in self.mfa_en_dict:
                    self.mfa_en_dict[wrd] = phns

    def get_dac_statistics(self):
        # (-0.0350, 2.6780)
        val_dl = self.model._validation_dl
        sample_cnt = 0

        mean = 0
        count = 0
        for batch in tqdm(val_dl):
            batch = self.model.transfer_batch_to_device(batch, self.model.device, 0)
            batch = self.model.parse_val_vb_input(batch)

            ori_mel = batch["mel"]
            ori_mel_lens = batch['mel_lens']
            self_attn_mask = batch["self_attn_mask"]
            self_attn_mask = rearrange(self_attn_mask, '... -> ... 1')

            sample_cnt += ori_mel.shape[0]
            ori_mel = ori_mel.masked_fill(~self_attn_mask, 0.)
            mean += ori_mel.sum()
            count += ori_mel.shape[-1] * ori_mel_lens.sum()
            if sample_cnt > 30000:
                break
        mean = mean / count
        
        sample_cnt = 0
        std = 0
        mean_std = 0
        count = 0
        cnt = 0
        for batch in tqdm(val_dl):
            batch = self.model.transfer_batch_to_device(batch, self.model.device, 0)
            batch = self.model.parse_val_vb_input(batch)

            ori_mel = batch["mel"]
            ori_mel_lens = batch['mel_lens']
            self_attn_mask = batch["self_attn_mask"]
            self_attn_mask = rearrange(self_attn_mask, '... -> ... 1')

            sample_cnt += ori_mel.shape[0]
            ori_mel = (ori_mel - mean) ** 2
            ori_mel = ori_mel.masked_fill(~self_attn_mask, 0.)
            mean_std += (ori_mel.sum() / (ori_mel.shape[-1] * ori_mel_lens.sum() - 1)) ** 0.5
            std += ori_mel.sum()
            count += ori_mel.shape[-1] * ori_mel_lens.sum()
            cnt += 1
            if sample_cnt > 30000:
                break
        std = (std / (count-1)) ** 0.5
        return mean, std

    def gen_v1_dataset_from_val_set(self, out_dir):
        """Generate masked-reconstructed dataset from validation set. No speech editing, could be used for deepfake detector training."""
        val_dl = self.model._validation_dl

        os.makedirs(f"{out_dir}/combine", exist_ok=True)
        label = [[], []] #fake, real

        for batch in tqdm(val_dl):
            batch = self.model.transfer_batch_to_device(batch, self.model.device, 0)
            batch = self.model.parse_val_vb_input(batch)

            cuts = batch["cuts"]
            ori_audio = batch["audio"]
            ori_mel = batch["mel"]
            ori_mel_lens = batch['mel_lens']
            cond = batch["cond"]    # same as ori_mel
            cond_mask = batch["cond_mask"]
            aligned_tokens = batch["aligned_tokens"]
            self_attn_mask = batch["self_attn_mask"]

            cond_st_idx = torch.arange(cond.shape[1], 0, -1, device=self.model.device).reshape(1, -1) * cond_mask
            cond_st = cond_st_idx.argmax(dim=1)
            cond_ed_idx = torch.arange(cond.shape[1], device=self.model.device).reshape(1, -1) * cond_mask
            cond_ed = cond_ed_idx.argmax(dim=1) + 1

            assert cond.shape[0] == cond_mask.shape[0] and cond.shape[1] == cond_mask.shape[1], \
                f"{cond.shape}, {cond_mask.shape}, {self_attn_mask.shape}, {aligned_tokens.shape}"
            try:
                gen_audio = self.model.cfm_wrapper.sample(
                    cond=cond,
                    self_attn_mask=self_attn_mask.bool(),
                    aligned_phoneme_ids=aligned_tokens,
                    cond_mask=cond_mask.bool(),
                    steps=16,
                )
            except:
                print("X")
                continue
            ori_audio_lens = batch["audio_lens"]
            gen_audio_lens = torch.clamp(ori_audio_lens, max=gen_audio.shape[-1])
            ori_ls = ori_audio_lens / self.model.voicebox.audio_enc_dec.sampling_rate
            gen_ls = gen_audio_lens / self.model.voicebox.audio_enc_dec.sampling_rate
            gen_st = gen_ls * cond_st / ori_mel_lens
            gen_ed = gen_ls * cond_ed / ori_mel_lens

            for i in range(cond.shape[0]):
                new_id = '-'.join(cuts[i].id.split('/'))
                if cond_st[i] == 0:
                    label[0].append(f"dev_fake_{new_id} {gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                elif cond_ed[i] == ori_mel_lens[i]:
                    label[0].append(f"dev_fake_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F 0")
                else:
                    label[0].append(f"dev_fake_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                label[1].append(f"dev_real_{new_id} 0.00-{ori_ls[i]:.2f}-T 1")

                _ori_audio = ori_audio[i, :ori_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_real_{new_id}.wav", _ori_audio, self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
                _gen_audio = gen_audio[i, :gen_audio_lens[i]].cpu().numpy()
                sf.write(f"{out_dir}/combine/dev_fake_{new_id}.wav", _gen_audio, self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')

        with open(f"{out_dir}/dev_label.txt", 'w') as f:
            f.write('\n'.join(label[0]))
            f.write('\n')
            f.write('\n'.join(label[1]))

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
    def filter_alignments(alignments: list[jiwer.AlignmentChunk]):
        for ali in alignments:
            if ali.type == "substitute":
                return True
        return False

    def gen_edit_transcript_json(self, out_json_path):
        val_dl = self.model._validation_dl

        out_file = open(out_json_path, 'w')

        for batch in tqdm(val_dl):
            texts = batch["texts"]
            cuts = batch["cuts"]
            for cut, text in zip(cuts, texts):
                alignment = cut.supervisions[0].alignment
                alignment = fix_alignment(alignment)
                words = [ali.symbol for ali in alignment["words"] if ali.symbol != "<eps>"]
                phns = [ali.symbol for ali in alignment["phones"] if ali.symbol != "sil"]
                if "<unk>" in words:
                    continue
                if "spn" in phns:
                    continue
                out = {
                    "id": cut.id,
                    "sys_prompt": sys_prompt,
                    "prompt": text,
                }
                json.dump(out, out_file)
                out_file.write('\n')

        out_file.close()

    def load_gpt_json(self, json_filename, out_filename):
        mfa_en_dict = {}
        with open("/root/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict", 'r') as f:
            for line in tqdm(f):
                wrd, _, _, _, _, phns = line.strip().split('\t')
                if wrd not in mfa_en_dict:
                    mfa_en_dict[wrd] = phns

        from collections import Counter
        out_dict = {}
        with open(json_filename, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            line = json.loads(line)
            assert line['id'] not in out_dict, line
            ref = self.normalize_text(line['prompt'])
            hyp = self.normalize_text(line['gpt_text'])
            out = jiwer.process_words(ref, hyp)
            _ref = out.references[0]
            _hyp = out.hypotheses[0]

            if not self.filter_alignments(out.alignments[0]):
                continue

            alis = [ali for ali in out.alignments[0] if ali.type == "substitute"]
            ali = random.choice(alis)
            froms = [_ref[i] for i in range(ali.ref_start_idx, ali.ref_end_idx)]
            tos = [_hyp[i] for i in range(ali.hyp_start_idx, ali.hyp_end_idx)]
            to = random.choice(tos)
            try:
                if to in mfa_en_dict:
                    phns = mfa_en_dict[to].split(' ')
                else:
                    phns = os.popen(f"conda run -n aligner bash -c \"echo '{to}' | mfa g2p -n 1 - english_us_arpa - 2> /dev/null\"").read().split('\t')[1].strip().split(' ')
            except:
                continue
            out_dict[line['id']] = {
                "ref": ' '.join(_ref),
                "hyp": ' '.join(_hyp),
                "froms": ' '.join(froms),
                "tos": ' '.join(tos),
                "to": to,
                "to_phns": phns,
            }
        with open(out_filename, 'w') as fo:
            json.dump(out_dict, fo, indent=4)
        return out_dict

    def gen_v3_dataset_from_val_set(self, edit_dict, out_dir, ztts=False, redit=True):
        """Generate SINE dataset from validation set"""
        val_dl = self.model._validation_dl

        os.makedirs(f"{out_dir}/combine", exist_ok=True)
        label = {"edit": [], "resyn": [], "real": []} #fake, real

        files = {
            "real": open(f"{out_dir}/medium_real.txt", 'w'),
            "resyn": open(f"{out_dir}/medium_resyn.txt", 'w'),
            "edit": open(f"{out_dir}/medium_edit.txt", 'w'),
        }

        if ztts:
            label["cut_paste"] = []
            files["cut_paste"] = open(f"{out_dir}/medium_cut_paste.txt", 'w')
        if redit:
            label["redit"] = []
            files["redit"] = open(f"{out_dir}/medium_redit.txt", 'w')

        count = 0
        for batch in tqdm(val_dl):
            batch = self.model.transfer_batch_to_device(batch, self.model.device, 0)

            cuts = batch["cuts"]
            alignments = []
            edit_froms = []
            edit_tos = []
            indexes = []
            for i, c in tqdm(enumerate(cuts), leave=False):
                assert c.id in edit_dict
                alignment = c.supervisions[0].alignment
                froms = edit_dict[c.id]["froms"]
                if "to" in edit_dict[c.id] and "to_phns" in edit_dict[c.id]:
                    to = (edit_dict[c.id]["to"], edit_dict[c.id]["to_phns"])
                else:
                    tos = edit_dict[c.id]["tos"]
                    to = random.choice(tos.split(' '))
                mfa_wrds = [ali.symbol for ali in alignment["words"]]
                froms = [wrd for wrd in froms.split(' ') if wrd in mfa_wrds]
                if len(froms) > 0:
                    indexes.append(i)
                    alignments.append(alignment)
                    edit_froms.append(random.choice(froms))
                    edit_tos.append(to)
            indexes = torch.tensor(indexes, device=self.model.device)

            ori_audio = torch.index_select(batch["audio"], 0, indexes)
            ori_audio_lens = torch.index_select(batch["audio_lens"], 0, indexes)
            ori_texts = [t for i, t in enumerate(batch["texts"]) if i in indexes]

            try:
                pred = self.model.forward(
                    audio=ori_audio,
                    audio_lens=ori_audio_lens,
                    texts=ori_texts,
                    alignments=alignments,
                    edit_from=edit_froms,
                    edit_to=edit_tos,
                    steps=16,
                    sample_std=self.sample_std,
                    dp_scale=1.1,
                    mfa_en_dict=self.mfa_en_dict,
                    ztts=ztts,
                )
            except:
                print("X")
                continue

            gen_audio_lens = pred["new_audio_lens"]
            ori_ls = ori_audio_lens / self.model.voicebox.audio_enc_dec.sampling_rate
            gen_ls = gen_audio_lens / self.model.voicebox.audio_enc_dec.sampling_rate
            gen_st_idx = pred["new_cond_st_idx"]
            gen_ed_idx = pred["new_cond_ed_idx"]
            gen_st = gen_st_idx / self.model.voicebox.audio_enc_dec.sampling_rate
            gen_ed = gen_ed_idx / self.model.voicebox.audio_enc_dec.sampling_rate

            for i in range(gen_audio_lens.shape[0]):
                _i = indexes[i].item()
                new_id = '-'.join(cuts[_i].id.split('/'))

                edit_audio = pred["edit_audio"]
                resyn_audio = pred["resyn_audio"]
                _audio = {
                    "real": ori_audio[i, :ori_audio_lens[i]].cpu().numpy(),
                    "resyn": resyn_audio[i, :ori_audio_lens[i]].cpu().numpy(),
                    "edit": edit_audio[i, :gen_audio_lens[i]].cpu().numpy(),
                }
                if ztts:
                    cap_audio = pred["cap_audio"]
                    _audio["cut_paste"] = cap_audio[i, :gen_audio_lens[i]].cpu().numpy()
                if redit:
                    redit_audio = pred["redit_audio"]
                    _audio["redit"] = redit_audio[i, :gen_audio_lens[i]].cpu().numpy()

                for edit_type in ["edit", "cut_paste", "redit"]:
                    if edit_type in label:
                        if gen_st_idx[i] == 0:
                            label[edit_type].append(f"dev_{edit_type}_{new_id} {gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                        elif gen_ed_idx[i] == gen_audio_lens[i]:
                            label[edit_type].append(f"dev_{edit_type}_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F 0")
                        else:
                            label[edit_type].append(f"dev_{edit_type}_{new_id} 0.00-{gen_st[i]:.2f}-T/{gen_st[i]:.2f}-{gen_ed[i]:.2f}-F/{gen_ed[i]:.2f}-{gen_ls[i]:.2f}-T 0")
                        sf.write(f"{out_dir}/combine/dev_{edit_type}_{new_id}.wav", _audio[edit_type], self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
                        files[edit_type].write(label[edit_type][-1] + '\n')
                        files[edit_type].flush()

                for real_type in ["real", "resyn"]:
                    label[real_type].append(f"dev_{real_type}_{new_id} 0.00-{ori_ls[i]:.2f}-T 1")
                    sf.write(f"{out_dir}/combine/dev_{real_type}_{new_id}.wav", _audio[real_type], self.model.voicebox.audio_enc_dec.sampling_rate, format='WAV')
                    files[real_type].write(label[real_type][-1] + '\n')
                    files[real_type].flush()

        for t in files:
            files[t].close()
        return label

