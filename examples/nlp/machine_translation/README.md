# NMT Distillation


This is a guide explaining how to use NMT distillation, including Hinton-style, DistilBERT-style, sequence-level interpolation, and hybrid distillation.

# Hinton-style Distillation

For NMT, we minimize NLL on a parallel training set of N sentences.
: 

<img src="https://render.githubusercontent.com/render/math?math=\color{darkgray}e^{i \pi} = -1">

## Run student training script

Run

```
python enc_dec_nmt_distillation
```
