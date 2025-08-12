# Overview
This file describes the frame-stacking implementation in Magpie-TTS.

# Frame-stacking

## Overview
Frame-stacking is a technique that allows the Magpie-TTS base decoder to process multiple consecutive audio frames in a single forward pass. The goal is to accelerate inference by reducing the number of generation steps of the base decoder.

Frame-stacking is typically used in combination with a second, smaller "Local Transformer" (LT) decoder. In this two-stage approach:

1. The base decoder processes multiple frames at once, producing a single latent representation for each group (stack) of frames
2. The Local Transformer then generates the individual `frames * codebooks` tokens.

The Local Transformer is much faster than the base decoder, making this two-stage approach significantly faster than generating each frame with the base decoder. The speed improvement comes from two factors:
* **Fewer parameters**: The LT decoder is lightweight compared to the base decoder
* **Shorter sequences**: The LT decoder only attends to the current frame stack and the latent, not the entire frame sequence

Although the base decoder can generate audio codes directly without the Local Transformer, using the LT decoder is typically necessary to achieve high-quality synthesis when frame-stacking is enabled.

## Design and Implementation
* The `frame_stacking_factor` is the parameter that controls the number of frames to stack. The default is 1, which means no frame-stacking. We have tested values up to `4`.
* For each codebooks, we keep a separate embedding table for at each frame withing the stack. At the input to the decoder, the embeddings are averages across codebooks (as usual) and also frames within the stack.

## Limitations
This is still WIP with more work to be done. Specifically, the following are not yet implemented / tested:
* Online code extraction combined with frame-stacking.
* Alignment encoder with frame-stacking.
* CTC loss with frame-stacking.