# NMT Distillation


This is a guide explaining how to use NMT distillation, including Hinton-style, DistilBERT-style, sequence-level interpolation, and hybrid distillation.

# Hinton-style Distillation

For NMT, we minimize NLL on a parallel training set of N sentences.
: 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_{\text{NLL}}(\theta)=-\sum_{n=1}^N\log p(\bf y^{(n)}|\bf x^{(n)};\theta)\\
=-\sum_{n=1}^N\sum_{t=1}^{\mathcal{T}}\log p(y_t^{(n)}|y^{(n)}_{<t},\bf h_{t-1}^{(n)},\text{Att}(\text{Enc}(\bf x^{(n)}), \bf y_{<t}^{(n)}, \bf h_{t-1}^{(n)});\theta)" title="NLL loss" />


## Run student training script

Run

```
python enc_dec_nmt_distillation
```
