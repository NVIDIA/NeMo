import argparse
import json
import os
import pprint
import string

import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.tts.modules.fcd_metric import FrechetCodecDistance
from nemo.collections.tts.models import AudioCodecModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import scripts.magpietts.evalset_config as evalset_config
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

def find_sample_audios(audio_dir):
    file_list = []
    for f in os.listdir(audio_dir):
        if "predicted_audio" in f and f.endswith(".wav"):
            audio_number = int(f.split("_")[-1].split(".wav")[0])
            file_list.append((audio_number, os.path.join(audio_dir, f)))
    file_list.sort()
    file_list = [t[1] for t in file_list]
    return file_list

def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def process_text(input_text):
    # Convert text to lowercase
    lower_case_text = input_text.lower()

    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")

    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")

    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))

    return single_space_text

def transcribe_with_whisper(whisper_model, whisper_processor, audio_path, language, device):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    # Set the language task (optional, improves performance for specific languages)
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language) if language else None
    inputs = whisper_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    inputs = inputs.to(device)
    # Generate transcription
    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

    # Decode transcription
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    result = transcription[0]
    return result

def extract_embedding(model, extractor, audio_path, device, sv_model_type):
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

    if sv_model_type == "wavlm":
        inputs = extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            embeddings = model(inputs).embeddings
    else:  # Titanet
        with torch.no_grad():
            embeddings = model.get_embedding(audio_path).squeeze()

    return embeddings.squeeze()

def evaluate(manifest_path, audio_dir, generated_audio_dir, language="en", sv_model_type="titanet", asr_model_name="stt_en_conformer_transducer_large",
             codecmodel_path=None, predicted_codes=None, predicted_codes_lens=None):
    audio_file_lists = find_sample_audios(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)

    device = "cuda"

    if language == "en":
        if asr_model_name == "stt_en_conformer_transducer_large":
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
        elif asr_model_name == "nvidia/parakeet-ctc-0.6b":
            asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")

        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
        asr_model = asr_model.to(device)
        asr_model.eval()
    else:
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
        whisper_model = whisper_model.to(device)
        whisper_model.eval()

    if sv_model_type == "wavlm":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        speaker_verification_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device).eval()
    else:
        feature_extractor = None
        speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        speaker_verification_model = speaker_verification_model.to(device)
        speaker_verification_model.eval()

    speaker_verification_model_alternate = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_small')
    speaker_verification_model_alternate = speaker_verification_model_alternate.to(device)
    speaker_verification_model_alternate.eval()

    if codecmodel_path is not None:
        codec = AudioCodecModel.restore_from(codecmodel_path, strict=False)
        codec = codec.to(device)
        codec.eval()
        codec_feature_dim = codec.vector_quantizer.codebook_dim
        fcd_metric = FrechetCodecDistance(codec=codec, feature_dim=codec_feature_dim).to(device)
    else:
        print("No codec model provided, skipping FCD metric")
        fcd_metric = None

    filewise_metrics = []
    pred_texts = []
    gt_texts = []
    gt_audio_texts = []
    for ridx, record in enumerate(records):
        gt_audio_filepath = record['audio_filepath']
        context_audio_filepath = record.get('context_audio_filepath', None)
        if audio_dir is not None:
            gt_audio_filepath = os.path.join(audio_dir, gt_audio_filepath)
            if context_audio_filepath is not None:
                context_audio_filepath = os.path.join(audio_dir, context_audio_filepath)
            # Update the FCD metric for *real* codes
            if fcd_metric is not None:
                 fcd_metric.update_from_audio_file(gt_audio_filepath, True)

        pred_audio_filepath = audio_file_lists[ridx]

        try:
            if language == "en":
                with torch.no_grad():
                    # import ipdb; ipdb.set_trace()
                    pred_text = asr_model.transcribe([pred_audio_filepath])[0].text
                    pred_text = process_text(pred_text)
                    gt_audio_text = asr_model.transcribe([gt_audio_filepath])[0].text
                    gt_audio_text = process_text(gt_audio_text)
            else:
                pred_text = transcribe_with_whisper(whisper_model, whisper_processor, pred_audio_filepath, language, device)
                pred_text = process_text(pred_text)
                gt_audio_text = transcribe_with_whisper(whisper_model, whisper_processor, gt_audio_filepath, language, device)
                gt_audio_text = process_text(gt_audio_text)
        except Exception as e:
            print("Error during ASR: {}".format(e))
            pred_text = ""
            gt_audio_text = ""

        if 'normalized_text' in record:
            gt_text = process_text(record['normalized_text'])
        else:
            gt_text = process_text(record['text'])

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)

        print("{} GT Text:".format(ridx), gt_text)
        print("{} Pr Text:".format(ridx), pred_text)
        # Format cer and wer to 2 decimal places
        print("CER:", "{:.4f} | WER: {:.4f}".format(detailed_cer[0], detailed_wer[0]))

        pred_texts.append(pred_text)
        gt_texts.append(gt_text)
        gt_audio_texts.append(gt_audio_text)

        pred_context_ssim = 0.0
        gt_context_ssim = 0.0
        with torch.no_grad():
            gt_speaker_embedding = extract_embedding(speaker_verification_model, feature_extractor, gt_audio_filepath, device, sv_model_type)
            pred_speaker_embedding = extract_embedding(speaker_verification_model, feature_extractor, pred_audio_filepath, device, sv_model_type)
            pred_gt_ssim = torch.nn.functional.cosine_similarity(gt_speaker_embedding, pred_speaker_embedding, dim=0).item()

            gt_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(gt_audio_filepath).squeeze()
            pred_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(pred_audio_filepath).squeeze()
            pred_gt_ssim_alternate = torch.nn.functional.cosine_similarity(gt_speaker_embedding_alternate, pred_speaker_embedding_alternate, dim=0).item()

            if context_audio_filepath is not None:
                context_speaker_embedding = extract_embedding(speaker_verification_model, feature_extractor, context_audio_filepath, device, sv_model_type)
                context_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(context_audio_filepath).squeeze()

                pred_context_ssim = torch.nn.functional.cosine_similarity(pred_speaker_embedding, context_speaker_embedding, dim=0).item()
                gt_context_ssim = torch.nn.functional.cosine_similarity(gt_speaker_embedding, context_speaker_embedding, dim=0).item()

                pred_context_ssim_alternate = torch.nn.functional.cosine_similarity(pred_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0).item()
                gt_context_ssim_alternate = torch.nn.functional.cosine_similarity(gt_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0).item()

        # update FCD metric for all generated codes
        if fcd_metric is not None:
            for i in range(predicted_codes.shape[0]):
                fcd_metric.update_from_codes(predicted_codes[i:i+1], predicted_codes_lens[i:i+1], False)

        filewise_metrics.append({
            'gt_text': gt_text,
            'pred_text': pred_text,
            'gt_audio_text': gt_audio_text,
            'detailed_cer': detailed_cer,
            'detailed_wer': detailed_wer,
            'cer': detailed_cer[0],
            'wer': detailed_wer[0],
            'pred_gt_ssim': pred_gt_ssim,
            'pred_context_ssim': pred_context_ssim,
            'gt_context_ssim': gt_context_ssim,
            'pred_gt_ssim_alternate': pred_gt_ssim_alternate,
            'pred_context_ssim_alternate': pred_context_ssim_alternate,
            'gt_context_ssim_alternate': gt_context_ssim_alternate,
            'gt_audio_filepath': gt_audio_filepath,
            'pred_audio_filepath': pred_audio_filepath,
            'context_audio_filepath': context_audio_filepath
        })

    filewise_metrics_keys_to_save = ['cer', 'wer', 'pred_context_ssim', 'pred_text', 'gt_text', 'gt_audio_filepath', 'pred_audio_filepath', 'context_audio_filepath']
    filtered_filewise_metrics = []
    for m in filewise_metrics:
        filtered_filewise_metrics.append({k: m[k] for k in filewise_metrics_keys_to_save})

    # Sort filewise metrics by cer in reverse
    filewise_metrics.sort(key=lambda x: x['cer'], reverse=True)

    # compute frechet distance for the whole test set
    if fcd_metric is not None:
        fcd = fcd_metric.compute().cpu().item()
        fcd_metric.reset()
    else:
        fcd = 0.0

    avg_metrics = {}
    avg_metrics['cer_filewise_avg'] = sum([m['detailed_cer'][0] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['wer_filewise_avg'] = sum([m['detailed_wer'][0] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['cer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0]
    avg_metrics['wer_cumulative'] = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[0]
    avg_metrics['ssim_pred_gt_avg'] = sum([m['pred_gt_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_context_avg'] = sum([m['pred_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_gt_context_avg'] = sum([m['gt_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_gt_avg_alternate'] = sum([m['pred_gt_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_pred_context_avg_alternate'] = sum([m['pred_context_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics['ssim_gt_context_avg_alternate'] = sum([m['gt_context_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)
    avg_metrics["cer_gt_audio_cumulative"] = word_error_rate_detail(hypotheses=gt_audio_texts, references=gt_texts, use_cer=True)[0]
    avg_metrics["wer_gt_audio_cumulative"] = word_error_rate_detail(hypotheses=gt_audio_texts, references=gt_texts, use_cer=False)[0]
    avg_metrics["frechet_codec_distance"] = fcd

    pprint.pprint(avg_metrics)

    return avg_metrics, filewise_metrics

def main():
    # audio_dir="/datap/misc/Datasets/riva" \
    parser = argparse.ArgumentParser(description='Evaluate Generated Audio')
    parser.add_argument('--manifest_path', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--generated_audio_dir', type=str, default=None)
    parser.add_argument('--whisper_language', type=str, default="en")
    parser.add_argument('--evalset', type=str, default=None)
    args = parser.parse_args()

    if args.evalset is not None:
        dataset_meta_info = evalset_config.dataset_meta_info
        assert args.evalset in dataset_meta_info
        args.manifest_path = dataset_meta_info[args.evalset]['manifest_path']
        args.audio_dir = dataset_meta_info[args.evalset]['audio_dir']

    evaluate(args.manifest_path, args.audio_dir, args.generated_audio_dir, args.whisper_language, sv_model_type="wavlm", asr_model_name="nvidia/parakeet-ctc-0.6b")



if __name__ == "__main__":
    main()
