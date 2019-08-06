NEural MOdules (NEMO) toolkit is a high level, framework-agnostic API for building AI-powered applications.

Core Principles
---------------
NEMO is built around these principles:

* Neural Module is a block that computes a set of outputs from a set of inputs.
    * Think of it as an abstraction between a layer and a neural network.
    * Examples: pre-trained language model, acoustic model, encoder, decoder, etc.
* NEMO-based application is a graph of :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` objects connected together via **typed** *input* and *output* ports.
    * :class:`NmTensor<nemo.core.neural_types.NmTensor>` objects flow between modules from port to port.
    * *Lazy execution* model. First, user defines activation flow graph. Then, calls *action* (such as training). Actual computation happen only after action is called.
* Neural modules' input and output ports and :class:`NmTensor<nemo.core.neural_types.NmTensor>` objects have *Neural Type* - :class:`NeuralType<nemo.core.neural_types.NeuralType>`.
* :class:`NeuralType<nemo.core.neural_types.NeuralType>` captures expected (or actual) Tensor dimensions' semantics, dimensionality and more.
* Domain or module-specific *Callbacks* are used for evaluations and inspecting training performance.


Built to run on GPUs
--------------------
On a toolkit-level NeMo provides:

* Mixed-precision training using Tensor Cores on NVIDIA's Volta and Turing GPUs
* Distributed training
* Distributed evaluation
