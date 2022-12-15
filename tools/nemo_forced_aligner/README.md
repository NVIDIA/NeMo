# NeMo Forced Aligner (NFA)

A tool for doing Forced Alignment using Viterbi decoding of NeMo CTC-based models.

## Usage example 

```
python <path_to_NeMo>/NeMo/tools/nemo_forced_aligner/align.py \
        manifest_filepath=<path to manifest of utterances you want to align> \
        output_ctm_folder=<path to where your ctm files will be saved>
```

## How do I use NeMo Forced Aligner?
To use NFA, all you need to provide is a correct NeMo manifest (with `"audio_filepath"` and `"text"` fields).

Call the `align.py` script, specifying the parameters as follows:

* `manifest_filepath`: The path to the manifest of the data you want to align, containing `'audio_filepath'` and `'text'` fields. The audio filepaths need to be absolute paths.

* `output_ctm_folder`: The folder where to save CTM files containing the generated alignments. There will be one CTM file per utterance (ie one CTM file per line in the manifest). The files will be called `<output_ctm_folder>/<utt_id>.ctm` and each line in each file will start with `<utt_id>`. By default, `utt_id` will be the stem of the audio_filepath. This can be changed by overriding `n_parts_for_ctm_id`.

* **[OPTIONAL]** `pretrained_name`: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded from NGC and used for generating the log-probs which we will use to do alignment. Any Quartznet, Citrinet, Conformer CTC model should work, in any language (only English has been tested so far). (Default: "stt_en_citrinet_1024_gamma_0_25").
>Note: NFA can only use CTC models (not Transducer models) at the moment. If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely give Out Of Memory errors.

* **[OPTIONAL]** `model_path`: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the log-probs which we will use to do alignment.
>Note: NFA can only use CTC models (not Transducer models) at the moment. If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely give Out Of Memory errors.

* **[OPTIONAL]** `model_downsample_factor`: the downsample factor of the ASR model. It should be 2 if your model is QuartzNet, 4 if it is Conformer CTC, 8 if it is Citrinet (Default: 8, to match with the default model_name "stt_en_citrinet_1024_gamma_0_25").

* **[OPTIONAL]** `separator`: the string used to separate CTM segments. If the separator is `“”` (empty string), the CTM segments will be the tokens used by the ASR model. If the separator is anything else, e.g. `“ “`, `“|”` or `“<new section>”`, the segments will be the blocks of text separated by that separator. (Default: `“ “`, so for languages such as English, the CTM segments will be words.)

* **[OPTIONAL]** `n_parts_for_ctm_id`: This specifies how many of the 'parts' of the audio_filepath we will use (starting from the final part of the audio_filepath) to determine the utt_id that will be used in the CTM files. (Default: 1, i.e. utt_id will be the stem of the basename of audio_filepath). Note also that any spaces that are present in the audio_filepath will be stripped away from the utt_id, so as not to change the number of space-separated elements in the CTM files.

* **[OPTIONAL]** `audio_sr`: The sample rate (in Hz) of your audio. (Default: 16000)

* **[OPTIONAL]** `device`: The device that will be used for generating log-probs and doing Viterbi decoding. (Default: 'cpu').

* **[OPTIONAL]** `batch_size`: The batch_size that will be used for generating log-probs and doing Viterbi decoding. (Default: 1).


# Input manifest file format
NFA needs to be provided with a 'manifest' file where each line specifies the absolute "audio_filepath" and "text" of each utterance that you wish to produce alignments for, like the format below:
```json
{"audio_filepath": "/absolute/path/to/audio.wav", "text": "the transcription of the utterance"}
```
> Note: NFA does not require "duration" fields, and can align long audio files without running out of memory. Depending on your machine specs, you can align audios up to 5-10 minutes on Conformer CTC models, up to around 1.5 hours for QuartzNet models, and up to several hours for Citrinet models. NFA will also produce better alignments the more accurate the ground-truth "text" is.


# Output CTM file format
For each utterance specified in a line of `manifest_filepath`, one CTM file will be generated at the location of `<output_ctm_folder>/<utt_id>.ctm`.
Each CTM file will contain lines of the format:
`<utt_id> 1 <start time in samples> <duration in samples> <word or token>`.
Note the second item in the line (the 'channel ID', which is required by the CTM file format) is always 1, as NFA operates on single channel audio.
