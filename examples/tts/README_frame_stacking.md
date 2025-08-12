# Overview
This file describes the frame-stacking implementation in Magpie-TTS.

# Frame-stacking

## Overview
Frame-stacking is a technique that allows the Magpie-TTSdecoder to process multiple audio with the aim of speeding up inference. It is usually paired with a second, "Local Transformer" (LT) decoder that predicts individual frames, but is much small and faster. To get good synthesis quality the LT decoder is generally needed

## Design and Implementation
* The `frame_stacking_factor` is a parameter that controls the number of frames to stack. The default is 1, which means no frame-stacking. We have tested values up to 4.
* We keep separate embeddings for codebooks at each stacking index. These embeddings are averaged at the input to the decoder.

## Limitations
The is still a work in progress with more work to be done. Specifically, the following are not yet implemented / tested:
* Online code extraction combined with frame-stacking.
* Alignment encoder with frame-stacking.
* CTC loss with frame-stacking.