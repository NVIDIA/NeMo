from nemo.collections.tts.models import T5TTS_Model
from nemo.collections.tts.data.text_to_speech_dataset import T5TTSDataset
from omegaconf.omegaconf import OmegaConf, open_dict
import os
import glob
import torch
import soundfile as sf
import evaluate_generated_audio
import evalset_config
import json
import argparse
import numpy as np
import scipy.stats as stats
import copy
import shutil
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from PIL import Image

def compute_mean_and_confidence_interval(metrics_list, metric_keys, confidence=0.90):
    metrics = {}
    for key in metric_keys:
        measurements = [m[key] for m in metrics_list]
        mean = np.mean(measurements)
        std_err = stats.sem(measurements)

        confidence_interval = std_err * stats.t.ppf((1 + confidence) / 2, len(measurements) - 1)
        print(f"{key}: {mean} +/- {confidence_interval}")
        metrics[key] = "{:.4f} +/- {:.4f}".format(mean, confidence_interval)
    return metrics

def run_inference(
        hparams_file, 
        checkpoint_file, 
        datasets, 
        out_dir, 
        temperature, 
        topk, 
        codecmodel_path, 
        use_cfg, 
        cfg_scale, 
        batch_size, 
        num_repeats=1, 
        apply_attention_prior=False,
        attention_prior_epsilon=1e-3,
        attention_prior_lookahead_window=10,
        estimate_alignment_from_layers=None,
        apply_prior_to_layers=None,
        start_prior_after_n_audio_steps=10,
        confidence_level=0.95
    ):
    # import ipdb; ipdb.set_trace()
    model_cfg = OmegaConf.load(hparams_file).cfg

    with open_dict(model_cfg):
        model_cfg.codecmodel_path = codecmodel_path
        if hasattr(model_cfg, 'text_tokenizer'):
            # Backward compatibility for models trained with absolute paths in text_tokenizer
            model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
            model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
            model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
        model_cfg.train_ds = None
        model_cfg.validation_ds = None


    model = T5TTS_Model(cfg=model_cfg)
    model.use_kv_cache_for_inference = True

    # Load weights from checkpoint file
    print("Loading weights from checkpoint")
    ckpt = torch.load(checkpoint_file)
    model.load_state_dict(ckpt['state_dict'])
    print("Loaded weights.")
    model.cuda()
    model.eval()
    # import ipdb; ipdb.set_trace()

    checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    checkpoint_name = "{}_Temp{}_Topk{}_Cfg_{}_{}_Prior_{}_{}_{}_start{}_Estlayers{}_PrLayers{}".format(
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
        "".join([str(l) for l in apply_prior_to_layers]) if apply_prior_to_layers is not None else "None"
    )
    dataset_meta_info = evalset_config.dataset_meta_info
    for dataset in datasets:
        metrics_n_repeated = []
        manifest_records = read_manifest(dataset_meta_info[dataset]['manifest_path'])
        for repeat_idx in range(num_repeats):
            eval_dir = os.path.join(out_dir, "{}_{}".format(checkpoint_name, dataset))
            audio_dir = os.path.join(eval_dir, "audio")
            pred_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(pred_audio_dir, exist_ok=True)
            language = dataset_meta_info[dataset].get('whisper_language', 'en')
            dataset_meta_for_dl = copy.deepcopy(dataset_meta_info[dataset])
            for key in ["whisper_language", "load_cached_codes_if_available"]:
                if key in dataset_meta_for_dl:
                    del dataset_meta_for_dl[key]

            dataset_meta = {dataset: dataset_meta_for_dl}
            context_durration_min = model.cfg.get('context_duration_min', 5.0)
            context_durration_max = model.cfg.get('context_duration_max', 5.0)
            if context_durration_min < 5.0 and context_durration_max > 5.0:
                context_durration_min = 5.0
                context_durration_max = 5.0 # @pneekhara - For multiencoder models, I want fixed size contexts for fair eval. Not too important though.
            test_dataset = T5TTSDataset(
                dataset_meta=dataset_meta,
                sample_rate=model_cfg.sample_rate,
                min_duration=0.5,
                max_duration=20,
                codec_model_downsample_factor=model_cfg.codec_model_downsample_factor,
                bos_id=model.bos_id,
                eos_id=model.eos_id,
                context_audio_bos_id=model.context_audio_bos_id,
                context_audio_eos_id=model.context_audio_eos_id,
                audio_bos_id=model.audio_bos_id,
                audio_eos_id=model.audio_eos_id,
                num_audio_codebooks=model_cfg.num_audio_codebooks,
                prior_scaling_factor=None,
                load_cached_codes_if_available=dataset_meta_info[dataset].get('load_cached_codes_if_available', True),
                dataset_type='test',
                tokenizer_config=None,
                load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
                use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
                pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
                context_duration_min=context_durration_min,
                context_duration_max=context_durration_max,
            )
            assert len(test_dataset) == len(manifest_records), "Dataset length and manifest length should be the same. Dataset length: {}, Manifest length: {}".format(len(test_dataset), len(manifest_records))
            test_dataset.text_tokenizer, test_dataset.text_conditioning_tokenizer = model._setup_tokenizers(model.cfg, mode='test')

            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=test_dataset.collate_fn,
                num_workers=2,
                shuffle=False
            )

            item_idx = 0
            for bidx, batch in enumerate(test_data_loader):
                print("Processing batch {} out of {} of dataset {}".format(bidx, len(test_data_loader), dataset))
                batch_cuda ={}
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch_cuda[key] = batch[key].cuda()
                    else:
                        batch_cuda[key] = batch[key]
                
                import time
                st = time.time()
                predicted_audio, predicted_audio_lens, _, _, cross_attention_maps, _  = model.infer_batch(
                    batch_cuda, 
                    max_decoder_steps=440, 
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
                    start_prior_after_n_audio_steps=start_prior_after_n_audio_steps
                )
                
                et = time.time()
                print(f"Time taken for inference: {et-st}", predicted_audio.size())
                for idx in range(predicted_audio.size(0)):
                    cross_attn_map_image = Image.fromarray(cross_attention_maps[idx])
                    cross_attn_map_image.save(os.path.join(audio_dir, f"cross_attn_map_{item_idx}.png"))

                    predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                    predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                    audio_path = os.path.join(pred_audio_dir, f"predicted_audio_{item_idx}.wav")
                    sf.write(audio_path, predicted_audio_np, model.cfg.sample_rate)
                    context_audio_path = manifest_records[item_idx].get('context_audio_filepath', None)
                    target_audio_path = manifest_records[item_idx].get('audio_filepath', None)
                    if context_audio_path is not None:
                        context_audio_path = os.path.join(dataset_meta_info[dataset]['audio_dir'], context_audio_path)
                    if target_audio_path is not None:
                        target_audio_path = os.path.join(dataset_meta_info[dataset]['audio_dir'], target_audio_path)
                    if os.path.exists(context_audio_path):
                        shutil.copy(context_audio_path, os.path.join(audio_dir, f"context_audio_{item_idx}.wav"))
                    if os.path.exists(target_audio_path):
                        shutil.copy(target_audio_path, os.path.join(audio_dir, f"target_audio_{item_idx}.wav"))
                    item_idx += 1
            
            metrics, filewise_metrics = evaluate_generated_audio.evaluate(
                dataset_meta[dataset]['manifest_path'],
                dataset_meta[dataset]['audio_dir'],
                pred_audio_dir,
                language=language,
            )
            metrics_n_repeated.append(metrics)
            with open(os.path.join(eval_dir, f"{dataset}_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            
            with open(os.path.join(eval_dir, f"{dataset}_filewise_metrics_{repeat_idx}.json"), "w") as f:
                # Indent for better readability
                json.dump(filewise_metrics, f, indent=4)

            all_experiment_csv = os.path.join(out_dir, "all_experiment_metrics.csv")
            if not os.path.exists(all_experiment_csv):
                with open(all_experiment_csv, "w") as f:
                    f.write("checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative\n")
            with open(all_experiment_csv, "a") as f:
                f.write(f"{checkpoint_name},{dataset},{metrics['cer_filewise_avg']},{metrics['wer_filewise_avg']},{metrics['cer_cumulative']},{metrics['wer_cumulative']},{metrics['ssim_pred_gt_avg']},{metrics['ssim_pred_context_avg']},{metrics['ssim_gt_context_avg']},{metrics['ssim_pred_gt_avg_alternate']},{metrics['ssim_pred_context_avg_alternate']},{metrics['ssim_gt_context_avg_alternate']},{metrics['cer_gt_audio_cumulative']},{metrics['wer_gt_audio_cumulative']}\n")
                print(f"Wrote metrics for {checkpoint_name} and {dataset} to {all_experiment_csv}")

        metric_keys = ['cer_filewise_avg', 'wer_filewise_avg', 'cer_cumulative', 'wer_cumulative', 
                       'ssim_pred_gt_avg', 'ssim_pred_context_avg', 'ssim_gt_context_avg', 
                       'ssim_pred_gt_avg_alternate', 'ssim_pred_context_avg_alternate', 'ssim_gt_context_avg_alternate',
                       'cer_gt_audio_cumulative', 'wer_gt_audio_cumulative'
                       ]
        metrics_mean_ci = compute_mean_and_confidence_interval(metrics_n_repeated, metric_keys, confidence=confidence_level)
        all_experiment_csv_with_ci = os.path.join(out_dir, "all_experiment_metrics_with_ci.csv")
        if not os.path.exists(all_experiment_csv_with_ci):
            with open(all_experiment_csv_with_ci, "w") as f:
                f.write("checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative\n")
        with open(all_experiment_csv_with_ci, "a") as f:
            f.write(f"{checkpoint_name},{dataset},{metrics_mean_ci['cer_filewise_avg']},{metrics_mean_ci['wer_filewise_avg']},{metrics_mean_ci['cer_cumulative']},{metrics_mean_ci['wer_cumulative']},{metrics_mean_ci['ssim_pred_gt_avg']},{metrics_mean_ci['ssim_pred_context_avg']},{metrics_mean_ci['ssim_gt_context_avg']},{metrics_mean_ci['ssim_pred_gt_avg_alternate']},{metrics_mean_ci['ssim_pred_context_avg_alternate']},{metrics_mean_ci['ssim_gt_context_avg_alternate']},{metrics_mean_ci['cer_gt_audio_cumulative']},{metrics_mean_ci['wer_gt_audio_cumulative']}\n")
            print(f"Wrote metrics with CI for {checkpoint_name} and {dataset} to {all_experiment_csv_with_ci}")


def main():
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--hparams_files', type=str, default="/datap/misc/continuouscheckpoints_ks3ks3/multiencoder_small_sp_ks3_hparams.yaml,/datap/misc/continuouscheckpoints_ks3ks3/decodercontext_small_sp_ks3Correct_hparams.yaml")
    parser.add_argument('--checkpoint_files', type=str, default="/datap/misc/continuouscheckpoints_ks3ks3/multiencoder_small_sp_ks3_epoch302.ckpt,/datap/misc/continuouscheckpoints_ks3ks3/decodercontext_small_sp_ks3Correct_epoch305.ckpt")
    parser.add_argument('--codecmodel_path', type=str, default="/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo")
    parser.add_argument('--datasets', type=str, default="libri_seen_test,libri_unseen_test")
    parser.add_argument('--base_exp_dir', type=str, default="/datap/misc/eosmount4/AllKernselSize3/NewTransformer")
    parser.add_argument('--draco_exp_dir', type=str, default="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos/experiments/NewT5TTS_FixedPosEmb/AllKernselSize3/NewTransformer")
    parser.add_argument('--server_address', type=str, default="pneekhara@login-eos02.eos.clusters.nvidia.com")
    parser.add_argument('--exp_names', type=str, default="multiencoder_small_sp_ks3_lnormapplied")
    parser.add_argument('--local_ckpt_dir', type=str, default="/datap/misc/continuouscheckpoints_fixedposembrough")
    parser.add_argument('--out_dir', type=str, default="/datap/misc/ContinuousEvalResults/NewTransformerKoelTTS")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--apply_attention_prior', action='store_true')
    parser.add_argument('--attention_prior_epsilon', type=float, default=1e-3)
    parser.add_argument('--attention_prior_lookahead_window', type=int, default=10)
    parser.add_argument('--estimate_alignment_from_layers', type=str, default=None)
    parser.add_argument('--apply_prior_to_layers', type=str, default=None)
    parser.add_argument('--start_prior_after_n_audio_steps', type=int, default=10)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--confidence_level', type=float, default=0.95)
    args = parser.parse_args()

    estimate_alignment_from_layers = None
    if args.estimate_alignment_from_layers is not None:
        estimate_alignment_from_layers = [int(l.strip()) for l in args.estimate_alignment_from_layers.split(",")]
    apply_prior_to_layers = None
    if args.apply_prior_to_layers is not None:
        apply_prior_to_layers = [int(l.strip()) for l in args.apply_prior_to_layers.split(",")]

    if (args.hparams_files is not None) and (args.checkpoint_files is not None) and (args.hparams_files != "null"):
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")
        print("Running inference for hparams files: ", hparam_files)
        print("Running inference for checkpoint files: ", checkpoint_files)
        assert len(hparam_files) == len(checkpoint_files), "Number of hparams files and checkpoint files should be the same."
        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            run_inference(
                hparams_file=hparams_file, 
                checkpoint_file=checkpoint_file,
                datasets=args.datasets.split(","),
                out_dir=args.out_dir,
                temperature=args.temperature,
                topk=args.topk,
                codecmodel_path=args.codecmodel_path,
                use_cfg=args.use_cfg,
                cfg_scale=args.cfg_scale,
                batch_size=args.batch_size,
                num_repeats=args.num_repeats,
                apply_attention_prior=args.apply_attention_prior,
                attention_prior_epsilon=args.attention_prior_epsilon,
                attention_prior_lookahead_window=args.attention_prior_lookahead_window,
                estimate_alignment_from_layers=estimate_alignment_from_layers,
                apply_prior_to_layers=apply_prior_to_layers,
                start_prior_after_n_audio_steps=args.start_prior_after_n_audio_steps,
                confidence_level=args.confidence_level,
            )
        return
    else:
        BASE_EXP_DIR = args.base_exp_dir
        DRACO_EXP_DIR = args.draco_exp_dir
        # Mount DRACO_EXP_DIR to BASE_EXP_DIR as follows:
        # sshfs -o allow_other pneekhara@draco-oci-dc-02.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/experiments/NewT5AllFixedFresh /datap/misc/dracomount/
        if args.exp_names is None:
            exp_names = os.listdir(BASE_EXP_DIR)
        else:
            exp_names = args.exp_names.split(",")

        for exp_name in exp_names:
            exp_dir = os.path.join(BASE_EXP_DIR, exp_name)
            # recurisvely look for hparams.yaml
            try:
                hparams_file = glob.glob(f"{exp_dir}/**/hparams.yaml", recursive=True)[0]
                checkpoints_dir = glob.glob(f"{exp_dir}/**/checkpoints", recursive=True)[0]
                last_checkpoint = (glob.glob(f"{checkpoints_dir}/*last.ckpt"))[0]
            except:
                print(f"Skipping experiment {exp_name} as hparams or last checkpoint not found.")
                continue
            last_checkpoint_path_draco = last_checkpoint.replace(BASE_EXP_DIR, DRACO_EXP_DIR) 
            epoch_num = last_checkpoint.split("epoch=")[1].split("-")[0]

            checkpoint_copy_path = os.path.join(args.local_ckpt_dir, f"{exp_name}_epoch_{epoch_num}.ckpt")
            hparams_copy_path = os.path.join(args.local_ckpt_dir, f"{exp_name}_hparams.yaml")
            
            scp_command = f"scp {args.server_address}:{last_checkpoint_path_draco} {checkpoint_copy_path}"
            print(f"Running command: {scp_command}")
            os.system(scp_command)
            print("Copied checkpoint.")
            hparams_path_draco = hparams_file.replace(BASE_EXP_DIR, DRACO_EXP_DIR)
            scp_command_hparams = f"scp {args.server_address}:{hparams_path_draco} {hparams_copy_path}"
            print(f"Running command: {scp_command_hparams}")
            os.system(scp_command_hparams)
            print("Copied hparams file.")
            # import ipdb; ipdb.set_trace()
            print("Hparams file path: ", hparams_copy_path)
            print("Checkpoint file path: ", checkpoint_copy_path)
            run_inference(
                hparams_copy_path, 
                checkpoint_copy_path, 
                datasets=args.datasets.split(","),
                out_dir=args.out_dir,
                temperature=args.temperature,
                topk=args.topk,
                codecmodel_path=args.codecmodel_path,
                use_cfg=args.use_cfg,
                cfg_scale=args.cfg_scale,
                batch_size=args.batch_size,
                num_repeats=args.num_repeats,
                apply_attention_prior=args.apply_attention_prior,
                attention_prior_epsilon=args.attention_prior_epsilon,
                attention_prior_lookahead_window=args.attention_prior_lookahead_window,
                estimate_alignment_from_layers=estimate_alignment_from_layers,
                apply_prior_to_layers=apply_prior_to_layers,
                start_prior_after_n_audio_steps=args.start_prior_after_n_audio_steps,
                confidence_level=args.confidence_level,
            )
            

if __name__ == '__main__':
    main()