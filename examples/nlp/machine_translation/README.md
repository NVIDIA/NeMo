# NMT Distillation


This is a guide explaining how to use NMT distillation, including Hinton-style, DistilBERT-style, sequence-level interpolation, and hybrid distillation.

# Hinton-style Distillation

For NMT, we minimize NLL on a parallel training set of N sentences.
: 

<img src=
"https://render.githubusercontent.com/render/math?math=\color{gray}%5Chuge+%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cmathcal%7BL%7D_%7B%5Ctext%7BNLL%7D%7D%28%5Ctheta%29%26%3D-%5Csum_%7Bn%3D1%7D%5EN+%5Clog+p%28%5Cbf+y%5E%7B%28n%29%7D%7C%5Cbf+x%5E%7B%28n%29%7D%29%5C%5C%26%3D-%5Csum_%7Bn%3D1%7D%5EN%5Csum_%7Bt%3D1%7D%5ET%5Clog+p%28%5Cbf+y%5E%7B%28n%29%7D_t+%7C+%5Cbf+y_%7B%3Ct%7D%5E%7B%28n%29%7D%2Ch_%7Bt-1%7D%5E%7B%28n%29%7D%2C%5Ctext%7BAtt%7D%28%5Ctext%7BEnc%7D%28%5Cbf+x%5E%7B%28n%29%7D%29%2Cy%5E%7B%28n%29%7D_%7B%3Ct%7D%2Ch_%7Bt-1%7D%5E%7B%28n%29%7D%29%3B%5Ctheta%29%0A%5Cend%7Balign%2A%7D%0A" 
alt="\begin{align*}
\mathcal{L}_{\text{NLL}}(\theta)&=-\sum_{n=1}^N \log p(\bf y^{(n)}|\bf x^{(n)})\\&=-\sum_{n=1}^N\sum_{t=1}^T\log p(\bf y^{(n)}_t | \bf y_{<t}^{(n)},h_{t-1}^{(n)},\text{Att}(\text{Enc}(\bf x^{(n)}),y^{(n)}_{<t},h_{t-1}^{(n)});\theta)
\end{align*}
">

## Run student training script

Run

```
python enc_dec_nmt_distillation
```
