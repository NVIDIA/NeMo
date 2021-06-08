Exporting NeMo Models
===========

Exporting Models
------

Most of NeMo models can be 'exported' to ONNX or TorchScript to be deployed for inference in optimized execution environments, such as Jarvis or Triton Inference Server.  
Export interface is provided by ``Exportable`` mix-in class. If model extends ``Exportable``, it can be exported:

.. code-block:: Python

   from nemo.core.classes import ModelPT, Exportable
   # deriving from Exportable
   class MyExportableModel(ModelPT, Exportable):
   ...

   mymodel = MyExportableModel.from_pretrained(model_name="MyModelName")
   
   # exporting pre-trained model to ONNX file for deployment.	
   mymodel.export('mymodel.onnx', [options])


How to Use Model Export
----------
Here are the arguments for ``Exportable.export()``. In most cases, you should only supply name of the output file and use all defaults:
.. code-block:: Python
    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 13,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = False,
        use_dynamic_axes: bool = True,
        dynamic_axes=None,
        check_tolerance=0.01,
    ):

``output``, ``input_example``, ``output_example``, ``verbose``, ``export_params``, ``do_constant_folding``, ``keep_initializers_as_inputs``, ``onnx_opset_version``, ``set_eval`` : those options have the same semantics as in Pytorch's onnx.export() and jit.trace() functions and are passed through. Here's Torch documentation on ``onnx.export()`` : https://pytorch.org/docs/stable/onnx.html#functions

File extension of 'output' parameter determines export format : .onnx->ONNX, .pt or .ts -> TorchScript. If input_example is None, ``Exportable.input_example()`` is called.

TorchScript-specific: if ``try_script`` is True, ``export()`` will try ``jit.script()`` before ``jit.trace()``.
``check_trace`` arg is passed through to ``jit.trace()``.
ONNX-specific: if ``use_dynamic_axes`` is True, ``onnx.export()`` will be called with dynamic axes. If ``dynamic_axes`` is None, they will be inferred from the model's ``input_types`` definition (batch dimension is dynamic, and so is duration etc).

If ``check_trace`` is True, resulting ONNX will also be run on the data passed in ``input_example`` and results compared to ``output_example`` using ``check_tolerance`` arg. Note higher tolerance default.


How to Make Model Exportable
----------

If you are simply using NeMo models, the above example is all you need to know.
Now if you write your own models, here's a few things you need to take care of after extending ``Exportable``:

Exportable hooks and overrides
~~~~~~~~~~~~~~~~~~~~~~~~~

You should not normally need to override ``Exportable`` default methods. However, ``Exportable.export()`` relies on a assumptions certain methods are available in your class:

.. code-block:: Python

    @property
    def input_example(self) # => Tuple(input, [(input, ...], [Dict])
         """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples. 
	 """
This function should return a tuple of (normally) Tensors - one per each of model inputs (args to forward()). Last element may be a Dict to specify non-positional arguments by name, as per Torch export() convention : https://pytorch.org/docs/stable/onnx.html#using-dictionaries-to-handle-named-arguments-as-model-inputs. 
  (Note: ``Dict`` currently does not work with Torchscript ``trace()``).
.. code-block:: Python

    @property
    def input_types(self):
    @property
    def output_types(self):
    
Those are needed for inferring in/out names and dynamic axes. If your Model derives from ModulePT, those are already there. Another common scenario is that your Model contains one or more modules that process input and generate output. Then, you should override ``Exportable`` methods ``input_module()`` and ``output_module()`` to point to them, like in this example:

.. code-block:: Python

    @property
    def input_module(self):
        return self.fastpitch

    @property
    def output_module(self):
        return self.fastpitch

Your Model should also have export-friendly ``forward()`` method - that may mean different things for ONNX ant TS. For ONNX, you can't have forced named parameters w/o default, like ``forward(self, *, text)``. For TS, you should avoid None and use ``Optional`` instead. The criterias are highly volatile and may change with every PyTorch version, so it's trial-and-error process. Now there is also general issue that in many cases, forward() for inference may be simplified and even use less inputs/outputs. To address all that, ``Exportable`` looks for ``forward_for_export()`` method in your Model and uses that instead of ``forward()`` to export:

.. code-block:: Python
    # Uses forced named args, many default parameters. 
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=0,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
    ):
        # Passes through all self.fastpitch outputs
        return self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=pace,
            spec=spec,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
        )


    # Uses less inputs, no '*', returns less outputs:
    def forward_for_export(self, text):
        (
            spect,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
        ) = self.fastpitch(text=text)
        return spect, durs_predicted, log_durs_predicted, pitch_predicted

To keep consistency with input_types()/output_types(), are also those hooks in ``Exportable`` that let you exclude particular inputs/outputs from export process:

.. code-block:: Python

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set(["durs", "pitch", "speaker", "pace", "spec", "attn_prior", "mel_lens", "input_lens"])

    @property
    def disabled_deployment_output_names(self):


Another common requirement for Models being exported is to run certain net modifications for inference efficiency before exporting - like disabling masks in some convolutions or remove batch normalizations. A better style is to make those happen on ModelPT.eval() (and reversed on .train()), but it's not always feasible so the following hook is provided in ``Exportable`` to run those:

.. code-block:: Python

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
	# do graph modifications specific for this model
        replace_1D_2D = kwargs.get('replace_1D_2D', False)
        replace_for_export(self, replace_1D_2D)
	# call base method for common set of modifications 
	Exportable._prepare_for_export(self, **kwargs)


Exportable Model code
~~~~~~~~~~~~~~~~~~~~~~~~~

Most importantly, actual Torch code in your Model should be ONNX or TorchScript - compatible (ideally, both).
First thing - it has to be, well, in Torch - avoid bare Numpy or Python operands (https://pytorch.org/docs/stable/onnx.html#write-pytorch-model-in-torch-way).
Second - make your Model ``Exportable`` right away and add an export unit test, that would catch any operation/construct not supported in ONNX/TS, immediately.

Please refer to PyTorch documentation:
       - List of supported operators: https://pytorch.org/docs/stable/onnx.html#supported-operators
       - Tracing vs. scripting: https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting 
       - AlexNet Example:  https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx 

