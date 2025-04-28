# Background
Magpie-TTS uses special tokens like AUDIO_BOS and AUDIO_EOS for its operation. The indices of these tokens are after the audio codec tokens, at the end of the embedding table.

In April 2025 we changed the layout of the embedding table in a non-backwards compatible way:

## Old Layout
With the most common codec configuration (2016 codes), the layout used to look like this:
```
| Index   | Token Description    | Comments                                                                                                  |
|---------|----------------------|-----------------------------------------------------------------------------------------------------------|
| [0]     | Codec Token 0        |                                                                                                           |
| [1]     | Codec Token 1        |                                                                                                           |
| [2]     | Codec Token 2        |                                                                                                           |
| ...     | ...                  |                                                                                                           |
| [2015]  | Codec Token 2015     |                                                                                                           |
| [2016]  | <Unused>             |                                                                                                           |
| [2017]  | <Unused>             |                                                                                                           |
| [2018]  | <Unused>             |                                                                                                           |
| ...     |                      |                                                                                                           |
| [2044]  | Context Audio BOS    | if model_type == `decoder_context_tts`                                                                    |
| [2045]  | Context Audio EOS    | if model_type == `decoder_context_tts`                                                                    |
| [2046]  | Audio BOS            | also used for Context Audio BOS if model_type == `multi_encoder_context_tts` or `single_encoder_sv_tts`   |
| [2047]  | Audio EOS            | also used for Context Audio EOS if model_type == `multi_encoder_context_tts` or `single_encoder_sv_tts`   |
```

## New Layout```
The new layout for the same codec configuration is:
```
| Index   | Token Description    | Comments  |
---------------------------------------------|
| [0]     | Codec Token 0        |           |
| [1]     | Codec Token 1        |           |
| [2]     | Codec Token 2        |           |
| ...     | ...                  |           |
| [2015]  | Codec Token 2015     |           |
| [2016]  | Audio BOS            |           |
| [2017]  | Audio EOS            |           |
| [2018]  | Context Audio BOS    |           |
| [2019]  | Context Audio EOS    |           |
| [2020]  | MASK token (MaskGit) |           |
| [2021]  | RESERVED_1           |           |
| [2022]  | RESERVED_2           |           |
| [2023]  | RESERVED_3           |           |
```

# How to Train and Load a New Checkpoint
For new trainings and inference all configuration is automatic:
* The number of codebooks, codec codebooks size, and codec downsampling rate are all read from the codec checkpoint rather than configured in Magpie.
* The embedding table size is automatically set to codec_codebook_size + number_of_special_tokens (currently 2016+8=2024). There is no risk of accidentally stepping on codec tokens since the table sizes gets automatically sized with enough room for the special tokens.

# How to Load Old Checkpoints
For checkpoints created before the change you can force legacy codebook layout in one of these ways:

## If using `infer_and_evaluate.py`
Just set the `--legacy_codebooks` command line option. No need to update your YAML file – The script will automatically add the overrides.

## If using a Hydra command line
You have two options:
### Add these to your command line
```
# decoder context model
+model.forced_num_all_tokens_per_codebook=2048 +model.forced_audio_eos_id=2047 +model.forced_audio_bos_id=2046  +model.forced_context_audio_eos_id=2045 +model.forced_context_audio_bos_id=2044

# multi encoder context and any other model type
+model.forced_num_all_tokens_per_codebook=2048 +model.forced_audio_eos_id=2047 +model.forced_audio_bos_id=2046  +model.forced_context_audio_eos_id=2047 +model.forced_context_audio_bos_id=2046
```
# Or, add these overrides to your YAML file
```
forced_num_all_tokens_per_codebook: 2048
forced_audio_eos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -1}           # 2047
forced_audio_bos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -2}           # 2046

# Depending on the old model type, the context_audio_bos_id and context_audio_eos_id will be different (choose one of the pairs below)

# For `multi_encoder_context_tts`, `single_encoder_sv_tts`:
#forced_context_audio_eos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -1}  # 2047
#forced_context_audio_bos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -2}  # 2046

# For `decoder_context_tts` models:
#forced_context_audio_eos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -3}   # 2045
#forced_context_audio_bos_id: ${sum:${model.forced_num_all_tokens_per_codebook}, -4}   # 2044
```
