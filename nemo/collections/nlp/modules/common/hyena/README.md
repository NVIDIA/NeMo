## Required Dependencies for Hyena

### Single-Head Hyena

Single-head Hyena requires the FlashFFTConv library. Please follow the [installation instructions](https://github.com/HazyResearch/flash-fft-conv?tab=readme-ov-file#installation) in the FlashFFTConv repository.

### Multi-Head Hyena

**Note: Only 8 heads are supported at the moment, with sequence length limited up to 8192.**

For multi-head support, install the `fftconv` library from the [Safari repository](https://github.com/HazyResearch/safari/tree/main/csrc/fftconv). Run the following in a terminal:

```bash
git clone https://github.com/HazyResearch/safari.git
cd safari/csrc/fftconv
pip install .
```
