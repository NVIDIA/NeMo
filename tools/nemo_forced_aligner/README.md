# NeMo Forced Aligner (NFA)

A tool for doing Forced Alignment using Viterbi decoding of NeMo CTC-based models.

## Usage example 

``` bash
python <path_to_NeMo>/NeMo/tools/nemo_forced_aligner/align.py \
        pretrained_name="stt_en_citrinet_1024_gamma_0_25" \
        model_downsample_factor=8 \
        manifest_filepath=<path to manifest of utterances you want to align> \
        output_ctm_folder=<path to where your ctm files will be saved>
```

## How do I use NeMo Forced Aligner?
To use NFA, all you need to provide is a correct NeMo manifest (with `"audio_filepath"` and `"text"` fields).

Call the `align.py` script, specifying the parameters as follows:

* `pretrained_name`: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded from NGC and used for generating the log-probs which we will use to do alignment. Any Quartznet, Citrinet, Conformer CTC model should work, in any language (only English has been tested so far). If `model_path` is specified, `pretrained_name` must not be specified.
>Note: NFA can only use CTC models (not Transducer models) at the moment. If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely give Out Of Memory errors.

* `model_path`: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the log-probs which we will use to do alignment. If `pretrained_name` is specified, `model_path` must not be specified.
>Note: NFA can only use CTC models (not Transducer models) at the moment. If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely give Out Of Memory errors.

* `model_downsample_factor`: the downsample factor of the ASR model. It should be 2 if your model is QuartzNet, 4 if it is Conformer CTC, 8 if it is Citrinet.

* `manifest_filepath`: The path to the manifest of the data you want to align, containing `'audio_filepath'` and `'text'` fields. The audio filepaths need to be absolute paths.

* `output_ctm_folder`: The folder where to save CTM files containing the generated alignments. There will be one CTM file per utterance (ie one CTM file per line in the manifest). The files will be called `<output_ctm_folder>/<utt_id>.ctm` and each line in each file will start with `<utt_id>`. By default, `utt_id` will be the stem of the audio_filepath. This can be changed by overriding `n_parts_for_ctm_id`.

* **[OPTIONAL]** `ctm_grouping_separator`: the string used to separate CTM segments. If the separator is `“”` (empty string) or `None`, the CTM segments will be the tokens used by the ASR model. If the separator is anything else, e.g. `“ “`, `“|”` or `“<new section>”`, the segments will be the blocks of text separated by that separator. (Default: `“ “`, so for languages such as English, the CTM segments will be words.)
> Note: if you pass in a hydra override `ctm_grouping_separator=" "`, hydra will remove the whitespace, thus converting that `" "` to `""`. If you want to pass in a space, make sure to use `ctm_grouping_separator="\ "`, or just do not pass in this override, as the default value is `" "` anyway.

* **[OPTIONAL]** `n_parts_for_ctm_id`: This specifies how many of the 'parts' of the audio_filepath we will use (starting from the final part of the audio_filepath) to determine the utt_id that will be used in the CTM files. (Default: 1, i.e. utt_id will be the stem of the basename of audio_filepath). Note also that any spaces that are present in the audio_filepath will be replaced with dashes, so as not to change the number of space-separated elements in the CTM files.

* **[OPTIONAL]** `transcribe_device`: The device that will be used for generating log-probs (i.e. transcribing). (Default: 'cpu').

* **[OPTIONAL]** `viterbi_device`: The device that will be used for doing Viterbi decoding. (Default: 'cpu').

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


# How do I evaluate the alignment accuracy?
Ideally you would have some 'true' CTM files to compare with your generated CTM files. 

Alternatively (or additionally), you can visualize the quality of alignments using tools such as Gecko, which can play your audio file and display the predicted alignments at the same time. The Gecko tool requires you to upload an audio file and at least one CTM file. The Gecko tool can be accessed here: https://gong-io.github.io/gecko/. More information about the Gecko tool can be found on its Github page here: https://github.com/gong-io/gecko. 
