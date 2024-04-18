Stable Diffusion XL Int8 Quantization
=======================================

This example shows how to use Ammo to calibrate and quantize the UNet part of the SDXL. The UNet part typically consumes
>95% of the e2e Stable Diffusion latency.

We also provide instructions on deploying and running E2E SDXL pipeline
with Ammo quantized int8 UNet to generate images and measure latency on target GPUs.

To get started, it is required to have a pretrained SDXL checkpoint in `nemo` format. The example training configs are provided in NeMo,
which is located in `NeMo/examples/multimodal/text2img/stable_diffusion`.

Calibration
---------------
The first step is to run quantization script with default config, and finally the script will export the quantized unet to onnx file.
Here is the default config for `NeMo/examples/multimodal/text2img/stable_diffusion/sd_xl_quantize.py`.


.. code-block:: yaml
    quantize
      exp_name: nemo
      n_steps: 20          # number of inference steps
      format: 'int8'       # only int8 quantization is supported now
      percentile: 1.0      # Control quantization scaling factors (amax) collecting range, meaning that we will collect the minimum amax in the range of `(n_steps * percentile)` steps. Recommendation: 1.0
      batch_size: 1        # batch size calling sdxl inference pipeline during calibration
      calib_size: 32       # For SDXL, we recommend 32, 64 or 128
      quant_level: 2.5     #Which layers to be quantized, 1: `CNNs`, 2: `CNN + FFN`, 2.5: `CNN + FFN + QKV`, 3: `CNN + Linear`. Recommendation: 2, 2.5 and 3, depending on the requirements for image quality & speedup.
      alpha: 0.8           # A parameter in SmoothQuant, used for linear layers only. Recommendation: 0.8 for SDXL



Important Parameters
^^^^^^^^^^^^^^^^^^^^
- percentile: Control quantization scaling factors (amax) collecting range, meaning that we will collect the minimum amax in the range of (n_steps * percentile) steps. Recommendation: 1.0
- alpha: A parameter in SmoothQuant, used for linear layers only. Recommendation: 0.8 for SDXL, 1.0 for SD 1.5
- quant-level: Which layers to be quantized, 1: CNNs, 2: CNN + FFN, 2.5: CNN + FFN + QKV, 3: CNN + Linear. Recommendation: 2, 2.5 and 3, depending on the requirements for image quality & speedup.
- calib-size: For SDXL, we recommend 32, 64 or 128, for SD 1.5, set to 512 or 1024.


Build the TRT engine for the Quantized ONNX UNet
------------------------------------------------------------

.. code-block:: bash
    trtexec --onnx=./nemo_onnx/unet.onnx --shapes=x:8x4x128x128,timesteps:8,context:8x80x2048,y:8x2816 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=nemo_unet_xl.plan


Important Parameters
^^^^^^^^^^^^^^^^^^^^
Input shape has to be provided here when building TRT engine.
- x: Input image latent shape (B * C * H * W)
- context: Input text conditioning (B * S * hidden_dimention)
- y: Additional embedding (B * adm_in_channels)

Build End-to-end Stable Diffusion XL Pipeline with NeMo
-----------------------------------------------------------

We provide a script to build end to end TRT inference pipeline with NeMo backend, which is located at `NeMo/examples/multimodal/text2img/stable_diffusion/sd_xl_export.py`.

.. code-block:: yaml
    infer:
        out_path: sdxl_export
        width: 1024
        height: 1024
        batch_size: 2

    trt:
      static_batch: False
      min_batch_size: 1
      max_batch_size: 8

Important Parameters
^^^^^^^^^^^^^^^^^^^^
- out_path: Directory to save onnx file and TRT engine files
- width and height: Image resolution of inference output
- batch_size: Only used for dummy input generation and onnx sanity check
- {min,max}_batch_size: The input batch size of TRT engine along its dynamic axis


Run End-to-end Stable Diffusion XL TRT Pipeline
-----------------------------------------------------------

The inference script can be found at `NeMo/examples/multimodal/text2img/stable_diffusion/sd_xl_trt_inference.py`.

.. code-block:: yaml
    unet_xl: sdxl_export/plan/unet_xl.plan
    vae: sdxl_export/plan/vae.plan
    clip1: sdxl_export/plan/clip1.plan
    clip2: sdxl_export/plan/clip2.plan

    out_path: trt_output


Please specify unet_xl as the quantized Unet engine to run the quantized solution. The system will load the original engine file by default.