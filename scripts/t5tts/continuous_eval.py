from nemo.collections.tts.models import T5TTS_Model
from nemo.collections.tts.data.text_to_speech_dataset import T5TTSDataset
from omegaconf.omegaconf import OmegaConf, open_dict
import os
import glob
import shutil
import torch
import soundfile as sf
import evaluate_generated_audio
import json
import argparse

dataset_meta_info = {
    'vctk': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths.json',
        'audio_dir' : '/datap/misc/Datasets/VCTK-Corpus',
        'feature_dir' : '/datap/misc/Datasets/VCTK-Corpus',
    },
    'riva_challenging': {
        'manifest_path' : '/home/pneekhara/2023/SimpleT5NeMo/manifests/challengingLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json',
        'audio_dir' : '/datap/misc/Datasets/riva',
        'feature_dir' : '/datap/misc/Datasets/riva',
    }
}


def run_inference(hparams_file, checkpoint_file, datasets, out_dir):
    model_cfg = OmegaConf.load(hparams_file).cfg

    with open_dict(model_cfg):
        model_cfg.codecmodel_path = "/datap/misc/checkpoints/AudioCodec_21Hz_no_eliz.nemo"
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
        model_cfg.train_ds = None
        model_cfg.validation_ds = None


    model = T5TTS_Model(cfg=model_cfg)

    # Load weights from checkpoint file
    print("Loading weights from checkpoint")
    ckpt = torch.load(checkpoint_file)
    model.load_state_dict(ckpt['state_dict'])
    print("Loaded weights.")
    model.cuda()
    model.eval()
    # import ipdb; ipdb.set_trace()

    checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    
    for dataset in datasets:
        eval_dir = os.path.join(out_dir, "{}_{}".format(checkpoint_name, dataset))
        audio_dir = os.path.join(eval_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True) 
        dataset_meta = {dataset: dataset_meta_info[dataset]}
        test_dataset = T5TTSDataset(
            dataset_meta=dataset_meta,
            sample_rate=model_cfg.sample_rate,
            min_duration=0.5,
            max_duration=20,
            codec_model_downsample_factor=model_cfg.codec_model_downsample_factor,
            bos_id=model.bos_id,
            eos_id=model.eos_id,
            audio_bos_id=model.audio_bos_id,
            audio_eos_id=model.audio_eos_id,
            num_audio_codebooks=model_cfg.num_audio_codebooks,
            prior_scaling_factor=None,
            load_cached_codes_if_available=True,
            dataset_type='test',
            tokenizer_config=None,
            load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
            use_text_conditioning_tokenizer=model.use_text_conditioning_encoder
        )
        test_dataset.text_tokenizer = model.tokenizer
        if model.use_text_conditioning_encoder:
            test_dataset.text_conditioning_tokenizer = model.text_conditioning_tokenizer


        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
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
            
            predicted_audio, predicted_audio_lens, _, _ = model.infer_batch(batch_cuda, max_decoder_steps=500)
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                audio_path = os.path.join(audio_dir, f"predicted_audio_{item_idx}.wav")
                sf.write(audio_path, predicted_audio_np, model.cfg.sample_rate)
                item_idx += 1


        metrics = evaluate_generated_audio.evaluate(
            dataset_meta[dataset]['manifest_path'],
            dataset_meta[dataset]['audio_dir'],
            audio_dir
        )

        with open(os.path.join(eval_dir, f"{dataset}_metrics.json"), "w") as f:
            json.dump(metrics, f)

        # Keys in metrics:
        # metrics['cer_filewise_avg']
        # metrics['wer_filewise_avg']
        # metrics['speaker_similarity']
        # metrics['cer_cumulative']
        # metrics['wer_cumulative']

        all_experiment_csv = os.path.join(out_dir, "all_experiment_metrics.csv")
        if not os.path.exists(all_experiment_csv):
            with open(all_experiment_csv, "w") as f:
                f.write("checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,speaker_similarity\n")
        with open(all_experiment_csv, "a") as f:
            f.write(f"{checkpoint_name},{dataset},{metrics['cer_filewise_avg']},{metrics['wer_filewise_avg']},{metrics['cer_cumulative']},{metrics['wer_cumulative']},{metrics['speaker_similarity']}\n")
            print(f"Wrote metrics for {checkpoint_name} and {dataset} to {all_experiment_csv}")



def main():
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--hparams_file', type=str, default="/datap/misc/continuouscheckpoints/SingleEncoder_WithPriorCTC0.002_Rope10k_hparams.yaml")
    parser.add_argument('--checkpoint_file', type=str, default="/datap/misc/continuouscheckpoints/SingleEncoder_WithPriorCTC0.002_Rope10k_epoch_9.ckpt")
    parser.add_argument('--datasets', type=str, default="riva_challenging,vctk")
    parser.add_argument('--base_exp_dir', type=str, default="/datap/misc/dracomount")
    parser.add_argument('--draco_exp_dir', type=str, default="/lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/experiments/NewT5AllFixedFresh")
    parser.add_argument('--exp_names', type=str, default="SingleEncoder_WithPriorCTC0.002_Rope10k,SingleEncoderTextContext_WithPriorCTC0.002_Rope10k,MultiEncoderTextContext_WithPriorCTC_Rope10k_NoPerceiverCTC0.002_10kall,MultiEncoder_WithPriorCTC_Rope10k_WithPerceiverCTC0.002")
    parser.add_argument('--out_dir', type=str, default="/datap/misc/ContinuousEvalResultsNewT5")
    args = parser.parse_args()

    if (args.hparams_file is not None) and (args.checkpoint_file is not None) and (args.hparams_file != "null"):
        run_inference(args.hparams_file, args.checkpoint_file, args.datasets.split(","), args.out_dir)
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
            checkpoint_copy_path = f"/datap/misc/continuouscheckpoints/{exp_name}_epoch_{epoch_num}.ckpt"
            hparams_copy_path = f"/datap/misc/continuouscheckpoints/{exp_name}_hparams.yaml"
            
            scp_command = f"scp pneekhara@draco-oci-dc-02.draco-oci-iad.nvidia.com:{last_checkpoint_path_draco} {checkpoint_copy_path}"
            print(f"Running command: {scp_command}")
            os.system(scp_command)
            print("Copied checkpoint.")
            hparams_path_draco = hparams_file.replace(BASE_EXP_DIR, DRACO_EXP_DIR)
            scp_command_hparams = f"scp pneekhara@draco-oci-dc-02.draco-oci-iad.nvidia.com:{hparams_path_draco} {hparams_copy_path}"
            print(f"Running command: {scp_command_hparams}")
            os.system(scp_command_hparams)
            print("Copied hparams file.")
            # import ipdb; ipdb.set_trace()
            print("Hparams file path: ", hparams_copy_path)
            print("Checkpoint file path: ", checkpoint_copy_path)

            
            run_inference(hparams_copy_path, checkpoint_copy_path, args.datasets.split(","), args.out_dir)
            

if __name__ == '__main__':
    main()