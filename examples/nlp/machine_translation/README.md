# NMT Distillation


This is a guide explaining how to use NMT distillation, including Hinton-style, DistilBERT-style, sequence-level interpolation, and hybrid distillation.

# Hinton-style Distillation

For NMT, we minimize NLL on a parallel training set of N sentences.
: 

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}_{\text{NLL}}(\theta)=-\sum_{n=1}^N\log%20p(\bf%20y^{(n)}|\bf%20x^{(n)};\theta)\\%20=-\sum_{n=1}^N\sum_{t=1}^{\mathcal{T}}\log%20p(y_t^{(n)}|y^{(n)}_{%3Ct},\bf%20h_{t-1}^{(n)},\text{Att}(\text{Enc}(\bf%20x^{(n)}),%20\bf%20y_{%3Ct}^{(n)},%20\bf%20h_{t-1}^{(n)});\theta)) 


## Run student training script

Run

```
python enc_dec_nmt_distillation
```
