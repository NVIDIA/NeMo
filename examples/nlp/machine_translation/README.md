# NMT Distillation


This is a guide explaining how to use NMT distillation, including Hinton-style, DistilBERT-style, sequence-level interpolation, and hybrid distillation.

# Hinton-style Distillation

For NMT, we minimize NLL on a parallel training set of $N$ sentences: 

```math
\mathcal{L}_{\text{NLL}}(\theta)=-\sum_{n=1}^N \log p(\bf y^{(n)}|\bf x^{(n)};\theta)
```

## Run student training script

Run

```
python enc_dec_nmt_distillation
```
