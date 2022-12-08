# NeMo Forced Aligner (NFA)

A tool for doing Forced Alignment using Viterbi decoding of NeMo CTC-based models.

## How do I use NeMo Forced Aligner?
To use NFA, all you need to provide is a correct NeMo manifest (with 'audio_filepath' and 'text' fields).

Call the align function in align.py, specifying the parameters as follows:

* `model_name`: Any Quartznet, Citrinet, Conformer CTC model should work, in any language (only English has been tested so far). You can provide a pretrained model name or a path to a saved '.nemo' file.

> Note: If you want to transcribe a long audio file (longer than ~5-10 mins), do not use Conformer CTC model as that will likely OOM.

* `model_downsample_factor`: should be 2 if your model is QuartzNet, 4 if it is Conformer CTC, 8 if it is Citrinet.

* `manifest_filepath`: The path to the manifest of the data you want to align.

* `output_ctm_folder`: The folder where to save CTM files containing the generated alignments. There will be one CTM file per utterance (ie one CTM file per line in the manifest). The files will be called <output_ctm_folder>/<utt_id>.ctm. By default, `utt_id` will be the stem of the audio_filepath. This can be changed by overriding `utt_id_extractor_func`.

* `grouping_for_ctm`: A string, either 'word' or 'basetoken'. 'basetoken' can mean either a token or character, depending on the model used for alignment. If you select 'basetoken' - the code will output a CTM with alignments at the token/character level. If you select 'word' - the code will group the tokens/characters into words.

* [OPTIONAL] `utt_id_extractor_func`: The function mapping the audio_filepath to the utt_id that will be used to save CTM files.

* `audio_sr`: The sample rate of your audio

* `device`: The device where you want your ASR Model to be loaded into.

* `batch_size`: The batch_size you wish to use for transcription and alignment.


# Example script

```python
from align import align

if __name__ == "__main__":
    align(
        model_name="stt_en_citrinet_1024_gamma_0_25", 
        model_downsample_factor=8,
        manifest_filepath=<path to manifest of utterances you want to align>,
        output_ctm_folder=<path to where your ctm files will be saved>,
        grouping_for_ctm="word", # either "word" or "basetoken"
        audio_sr=16000, # the sampling rate of your audio files
        device="cuda:0", # the device that the ASR model will be loaded into 
        batch_size=1,
    )

```