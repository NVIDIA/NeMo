NeMo (**Ne**ural **Mo**dules) is a toolkit for creating AI applications built around **neural modules**, conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs.

**NeMo Core** provides the fundamental building blocks for all neural models and NeMo's type system.

Core Principles
---------------
NEMO is built around these principles:

* Neural Module is a block that computes a set of outputs from a set of inputs.
    * Think of it as an abstraction between a layer and a neural network.
    * Examples: pre-trained language model, acoustic model, encoder, decoder, etc.
* NeMo-based application is a DAG of :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` objects connected together via **typed** *input* and *output* ports.
    * :class:`NmTensor<nemo.core.neural_types.NmTensor>` objects flow between modules from port to port.
    * *Lazy execution* model. First, user defines activation flow graph. Then, calls *action* (such as training). Actual computation happen only after action is called.
* The input and output ports of each neural module and :class:`NmTensor<nemo.core.neural_types.NmTensor>` objects all have *Neural Type* - :class:`NeuralType<nemo.core.neural_types.NeuralType>`.
* :class:`NeuralType<nemo.core.neural_types.NeuralType>` the semantics, axis order, and dimensions of a tensor and more.
* Domain or module-specific *Callbacks* are used for evaluations and inspecting training performance.


Built to run on GPUs
--------------------
NeMo Core provides:

* Mixed-precision training using Tensor Cores on NVIDIA's Volta and Turing GPUs
* Distributed training
* Distributed evaluation
