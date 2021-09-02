# NMT Distillation


This is a guide explaining how to use NMT distillation, including Hinton-style, DistilBERT-style, sequence-level interpolation, and hybrid distillation.

# Hinton-style Distillation

For NMT, we minimize NLL on a parallel training set of N sentences.
: 

<img src="https://render.githubusercontent.com/render/math?math=\Huge\color{gray}\mathcal{L}_{\text{NLL}}(\theta)=-\sum_{n=1}^N \log p(\bf y^{(n)}|\bf x^{(n)})=-\sum_{n=1}^N\sum_{t=1}^T\log p(\bf y^{(n)}_t | \bf y_{<t}^{(n)},h_{t-1}^{(n)},\text{Att}(\text{Enc}(\bf x^{(n)}),y^{(n)}_{<t},h_{t-1}^{(n)});\theta)">

## Run student training script

Run

```
python enc_dec_nmt_distillation
```
