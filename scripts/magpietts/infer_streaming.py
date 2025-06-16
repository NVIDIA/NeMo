from omegaconf.omegaconf import OmegaConf, open_dict
import os
import torch
import soundfile as sf
import evaluate_generated_audio
import evalset_config
import numpy as np
import copy
import random
from PIL import Image

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers
from nemo.collections.tts.models import MagpieTTSModel

def update_config(model_cfg, codecmodel_path, legacy_codebooks=False):
    ''' helper function to rename older yamls from t5 to magpie '''
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        # Added to prevent crash after removing arg from transformer_2501.py in https://github.com/blisc/NeMo/pull/56
        del model_cfg.decoder.prior_eps
    if legacy_codebooks:
        # Added to address backward compatibility arising from
        #  https://github.com/blisc/NeMo/pull/64
        print("WARNING: Using legacy codebook indices for backward compatibility. Should only be used with old checkpoints.")
        num_audio_tokens_per_codebook = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens_per_codebook
        model_cfg.forced_audio_eos_id = num_audio_tokens_per_codebook - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens_per_codebook - 2
        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 4
            model_cfg.forced_mask_token_id = num_audio_tokens_per_codebook - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 2

    return model_cfg


def chunk_and_tokenize_text(text, text_chunk_size, num_chunk_per_window, tokenizer_name, text_tokenizer, eos_token_id):
    # print(f"text {text}")
    split_text = text.split() # []
    print(f"split_text {len(split_text)}")
    chunked_tokens = []
    chunked_tokens_len = []
    chunked_text = []
    current_text = " ".join(split_text[:(num_chunk_per_window*text_chunk_size)])
    print(f"s: 0 e: {num_chunk_per_window*text_chunk_size}")
    # print(f"current_text {current_text}")
    chunked_text.append(current_text)
    tokens = text_tokenizer.encode(text=current_text, tokenizer_name=tokenizer_name)
    tokens = torch.tensor(tokens, dtype=torch.int32)
    # print(f"tokens {tokens.shape}")
    tokens_len = tokens.shape[0]
    chunked_tokens.append(tokens)
    chunked_tokens_len.append(tokens_len)
    start = num_chunk_per_window*text_chunk_size
    for i in range(start, len(split_text), text_chunk_size):
        current_text = " ".join(split_text[i:min(i+text_chunk_size, len(split_text))])
        print(f"s: {i} e: {min(i+text_chunk_size, len(split_text))}")
        # print(f"current_text {current_text}")
        chunked_text.append(current_text)
        tokens = text_tokenizer.encode(text=current_text, tokenizer_name=tokenizer_name)
        if i+text_chunk_size >= len(split_text):
            tokens = tokens + [eos_token_id]
        tokens = torch.tensor(tokens, dtype=torch.int32)
        tokens_len = tokens.shape[0]
        # print(f"tokens {tokens.shape}")
        chunked_tokens.append(tokens)
        chunked_tokens_len.append(tokens_len)
    print(f"...token len {sum(chunked_tokens_len)}")

    return chunked_tokens, chunked_tokens_len, chunked_text

def run_inference_streaming(
        hparams_file, 
        checkpoint_file, 
        nemo_file,
        datasets, 
        out_dir, 
        temperature, 
        topk, 
        codecmodel_path, 
        use_cfg, 
        cfg_scale, 
        batch_size, 
        sv_model,
        asr_model_name,
        num_repeats=1, 
        apply_attention_prior=False,
        attention_prior_epsilon=1e-3,
        attention_prior_lookahead_window=10,
        estimate_alignment_from_layers=None,
        apply_prior_to_layers=None,
        start_prior_after_n_audio_steps=0,
        confidence_level=0.95,
        use_local_transformer=False,
        maskgit_n_steps=3,
        legacy_codebooks=False,
        use_exponential_weight=True,
        tokenizer_name=None,
        clean_up_disk=False,
    ):
    num_chunk_per_window = 2
    num_audio_tokens_per_text = 1
    true_window_size = 250
    text_chunk_size = 2
    # Load model
    if hparams_file is not None:
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        with open_dict(model_cfg):
            model_cfg = update_config(model_cfg, codecmodel_path, legacy_codebooks)

        model = MagpieTTSModel(cfg=model_cfg)
        text_tokenizer, text_conditioning_tokenizer = setup_tokenizers(model.cfg.text_tokenizers, model.cfg.use_text_conditioning_encoder, mode='test')
        model.use_kv_cache_for_inference = True

        # Load weights from checkpoint file
        print("Loading weights from checkpoint")
        ckpt = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    elif nemo_file is not None:
        model_cfg = MagpieTTSModel.restore_from(nemo_file, return_config=True)
        with open_dict(model_cfg):
            model_cfg = update_config(model_cfg, codecmodel_path)
        model = MagpieTTSModel.restore_from(nemo_file, override_config_path=model_cfg)
        text_tokenizer, text_conditioning_tokenizer = setup_tokenizers(model.cfg.text_tokenizers, model.cfg.use_text_conditioning_encoder, mode='test')
        model.use_kv_cache_for_inference = True
        checkpoint_name = nemo_file.split("/")[-1].split(".nemo")[0]
    else:
        raise ValueError("Need a checkpoint")

    print("Loaded weights.")
    model.cuda()
    model.eval()

    checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0] if checkpoint_file is not None else checkpoint_name
    checkpoint_name = "{}_Temp{}_Topk{}_Cfg_{}_{}_Prior_{}_{}_{}_start{}_Estlayers{}_PrLayers{}_LT_{}_sv_{}".format(
        checkpoint_name,
        temperature,
        topk,
        use_cfg,
        cfg_scale,
        apply_attention_prior,
        attention_prior_epsilon,
        attention_prior_lookahead_window,
        start_prior_after_n_audio_steps,
        "".join([str(l) for l in estimate_alignment_from_layers]) if estimate_alignment_from_layers is not None else "None",
        "".join([str(l) for l in apply_prior_to_layers]) if apply_prior_to_layers is not None else "None",
        use_local_transformer,
        sv_model
    )
    dataset_meta_info = evalset_config.dataset_meta_info
    for dataset in datasets:
        manifest_records = []
        _manifest_records = read_manifest(dataset_meta_info[dataset]['manifest_path'])
        for record in _manifest_records:
            if record['duration'] >= 0.5 and record['duration'] <= 20:
                manifest_records.append(record)
        for repeat_idx in range(num_repeats):
            eval_dir = os.path.join(out_dir, "{}_{}".format(checkpoint_name, dataset))
            audio_dir = os.path.join(eval_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            pred_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(pred_audio_dir, exist_ok=True)
            language = dataset_meta_info[dataset].get('whisper_language', 'en')
            dataset_meta_for_dl = copy.deepcopy(dataset_meta_info[dataset])
            for key in ["whisper_language", "load_cached_codes_if_available"]:
                if key in dataset_meta_for_dl:
                    del dataset_meta_for_dl[key]

            if tokenizer_name is not None:
                dataset_meta_for_dl['tokenizer_names'] = [tokenizer_name]

            dataset_meta = {dataset: dataset_meta_for_dl}
            context_duration_min = model.cfg.get('context_duration_min', 5.0)
            context_duration_max = model.cfg.get('context_duration_max', 5.0)
            codec_model_downsample_factor = model_cfg.codec_model_downsample_factor if "codec_model_downsample_factor" in model_cfg else model._codec_model.samples_per_frame
            sample_rate = model_cfg.sample_rate
            context_audio_bos_id=model.context_audio_bos_id,
            context_audio_eos_id=model.context_audio_eos_id,
            audio_bos_id=model.audio_bos_id,
            audio_eos_id=model.audio_eos_id,
            batch = {}

            print(f"manifest_records {len(manifest_records)}")
            for idx, entry in enumerate(manifest_records):
                if "context_audio_codes_path" not in entry:
                    ## TODO(sugh): Change this hard coding when Causal audio codec codes are available and part of manifest
                    # For 21FPS
                    ### Lindy
                    # entry['context_audio_codes_path'] = "/checkpoints/streaming/magpie/jason/LINDY_CMU_NEUTRAL_000508.pt"
                    ### Jensen
                    entry['context_audio_codes_path'] = "/checkpoints/streaming/magpie/jensen/GTC20_FALL_KEYNOTE-VOOnly-44khz-16bit-mono_652.pt"
                    # For 25FPS
                    # entry['context_audio_codes_path'] = "/checkpoints/streaming/magpie/jason/LINDY_CMU_NEUTRAL_000508.pt"
                if "normalized_text" in entry:
                    text = entry["normalized_text"]
                else:
                    text = entry["text"]

                chunked_tokens, chunked_tokens_len, chunked_text_list = chunk_and_tokenize_text(
                    text, 
                    text_chunk_size, 
                    num_chunk_per_window, 
                    "english_phoneme" if tokenizer_name is None else tokenizer_name, 
                    text_tokenizer, 
                    model.eos_id
                ) # List, List
                generate_chunk_audios = False # len(text.split()) > 1000
                print(f"...generate_chunk_audios {generate_chunk_audios}")

                assert 'context_audio_codes_path' in entry, f"Context audio codes path not found in manifest entry: {entry}"
                
                context_audio_codes_path = entry['context_audio_codes_path']
                context_audio_codes = torch.load(context_audio_codes_path).long() # (8, T)
                # Sample random duration between self.context_duration_min and self.context_duration_max
                _context_duration_to_slice = random.uniform(context_duration_min, context_duration_max)
                _num_frames_to_slice = int(_context_duration_to_slice * sample_rate / codec_model_downsample_factor)
                if _num_frames_to_slice < context_audio_codes.shape[1]:
                    start_idx = random.randint(0, context_audio_codes.shape[1] - _num_frames_to_slice)
                    context_audio_codes = context_audio_codes[:, start_idx:start_idx+_num_frames_to_slice]
                else:
                    # Repeaet the audio if it is shorter than the desired duration
                    _num_repeats = int(np.ceil(_num_frames_to_slice / context_audio_codes.shape[1]))
                    # context_audio_codes is a tensor of shape (num_codebooks, T)
                    context_audio_codes_repeated = context_audio_codes.repeat(1, _num_repeats)
                    context_audio_codes = context_audio_codes_repeated[:, :_num_frames_to_slice]
                print(f"context_audio_codes {context_audio_codes.shape}")


                context_bos_tensor = torch.full((context_audio_codes.shape[0], 1), context_audio_bos_id[0], dtype=context_audio_codes.dtype)
                context_eos_tensor = torch.full((context_audio_codes.shape[0], 1), context_audio_eos_id[0], dtype=context_audio_codes.dtype)
                context_audio_codes = torch.cat([context_bos_tensor, context_audio_codes, context_eos_tensor], dim=1)
                context_audio_codes_len = torch.tensor([context_audio_codes.shape[1]])
                context_audio_codes = context_audio_codes.unsqueeze(0)
                batch['context_audio_codes'] = context_audio_codes.cuda()
                batch['context_audio_codes_lens'] = context_audio_codes_len.cuda()
                batch['has_text_context'] = torch.BoolTensor([False]).cuda()
                # batch['context_text_tokens'] = torch.tensor(text_conditioning_tokenizer("[NO TEXT CONTEXT]")['input_ids'], dtype=torch.int32).cuda().unsqueeze(0)
                # context_text_len = [batch['context_text_tokens'].shape[1]]
                # batch['context_text_tokens_lens'] = torch.tensor(context_text_len).cuda()

                model.set_streaming_inference_variables(num_audio_tokens_per_text=num_audio_tokens_per_text, true_window_size=true_window_size)
                predicted_codes = []
                predicted_codes_lens = 0
                input_len = 0
                model.encoder.reset_cache(use_cache=False)
                model.decoder.reset_cache(use_cache=False)
                torch.cuda.empty_cache()
                for token_idx, inputs in enumerate(zip(chunked_tokens, chunked_tokens_len)):
                    current_tokens, current_tokens_lens = inputs
                    current_tokens = current_tokens.unsqueeze(0)
                    # print(f"current_tokens {current_tokens.shape}")
                    # print(f"current_tokens_lens {current_tokens_lens}")
                    # print(f"batch['context_audio_codes'] {batch['context_audio_codes'].shape}")
                    # print(f"batch['context_audio_codes_len'] {batch['context_audio_codes_lens']}")
                    batch['text'] = current_tokens.cuda()
                    batch['text_lens'] = torch.tensor([current_tokens_lens]).cuda()
                    input_len += current_tokens_lens
                    # print(f"batch['text_lens'] {batch['text_lens']}")
                    # print(f"token_idx {token_idx} chunked_text_list {chunked_text_list[token_idx]} current_tokens_lens {current_tokens_lens} input_len {input_len}")
                    is_end_of_text = token_idx == (len(chunked_tokens) - 1)
                    beginning_of_text = token_idx == 0
                    # model.encoder.reset_cache(use_cache=False)
                    model.decoder.reset_cache(use_cache=False)
                    current_predicted_codes, current_predicted_codes_lens, cross_attention_maps, _ = model.generate_speech_per_chunk_of_text(
                        batch, 
                        is_end_of_text, 
                        beginning_of_text,
                        max_decoder_steps=50000, 
                        temperature=temperature, 
                        topk=topk, 
                        use_cfg=use_cfg,
                        cfg_scale=cfg_scale, 
                        return_cross_attn_probs=True, 
                        apply_attention_prior=apply_attention_prior,
                        prior_epsilon=attention_prior_epsilon,
                        lookahead_window_size=attention_prior_lookahead_window,
                        estimate_alignment_from_layers=estimate_alignment_from_layers,
                        apply_prior_to_layers=apply_prior_to_layers,
                        start_prior_after_n_audio_steps=start_prior_after_n_audio_steps,
                        go_to_end_of_text_window=True,
                        use_exponential_weight=use_exponential_weight,
                    )
                    # print(f"current_predicted_audio {current_predicted_codes.shape}")
                    # print(f"current_predicted_audio_lens {current_predicted_codes_lens}")
                    predicted_codes.append(current_predicted_codes)
                    predicted_codes_lens += current_predicted_codes_lens[0]
                    # print(f"--->>> predicted_audio_lens {predicted_codes_lens}")

                    if generate_chunk_audios:
                        current_predicted_codes_lens = torch.tensor([current_predicted_codes_lens[0]]).long().cuda()
                        current_predicted_audio, current_predicted_audio_lens = model.codes_to_audio(current_predicted_codes, current_predicted_codes_lens)
                        # print(f"current_predicted_audio {current_predicted_audio.size()}")
                        if token_idx == 0:
                            concatenated_predicted_audio_np = current_predicted_audio.squeeze(0).float().detach().cpu().numpy()
                        else:
                            concatenated_predicted_audio_np = np.concatenate((concatenated_predicted_audio_np, current_predicted_audio.squeeze(0).float().detach().cpu().numpy()))
                        # print(f"concatenated_predicted_audio_np {concatenated_predicted_audio_np.shape}")

                    for maps in cross_attention_maps:
                        if generate_chunk_audios:
                            for batch_idx, map in enumerate(maps):
                                cross_attn_map_image = Image.fromarray(map)
                                cross_attn_map_image.save(os.path.join(audio_dir, f"cross_attn_map_{token_idx}_{batch_idx}.png"))
                torch.cuda.empty_cache()
                if generate_chunk_audios:
                    predicted_audio_np = concatenated_predicted_audio_np
                else:
                    predicted_codes = torch.cat(predicted_codes, dim=2).cuda()
                    print(f"...predicted_codes {predicted_codes.shape}")
                    print(f"...predicted_codes_lens {predicted_codes_lens}")
                    predicted_codes_lens = torch.tensor([predicted_codes_lens]).long().cuda()
                    predicted_audio, predicted_audio_lens = model.codes_to_audio(predicted_codes, predicted_codes_lens)
                    print(f"...predicted_audio {predicted_audio.shape}")
                    print(f"...predicted_audio_lens {predicted_audio_lens}")
                    predicted_audio_np = predicted_audio.squeeze(0).float().detach().cpu().numpy()
                    # predicted_audio_np = predicted_audio_np[:predicted_audio_lens]
                    print(f"...predicted_audio_np {predicted_audio_np.shape} model.cfg.sample_rate {model.cfg.sample_rate}")
                audio_path = os.path.join(pred_audio_dir, f"predicted_audio_{idx}.wav")
                sf.write(audio_path, predicted_audio_np, model.cfg.sample_rate)
            metrics, filewise_metrics = evaluate_generated_audio.evaluate(
                dataset_meta[dataset]['manifest_path'],
                dataset_meta[dataset]['audio_dir'],
                pred_audio_dir,
                language=language,
                sv_model_type=sv_model,
                asr_model_name=asr_model_name,
            )
            cer = metrics['cer_cumulative']
            ssim = metrics['ssim_pred_context_avg']
            print(f"metrics {metrics}")
            print(f"filewise_metrics {filewise_metrics}")
    if clean_up_disk:
        shutil.rmtree(out_dir)
    return cer, ssim
