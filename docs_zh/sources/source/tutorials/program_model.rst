Programming Model
-------------------

A typical application using NeMo APIs consists of 3 logical stages:

1) Creation of :class:`NeuralModuleFactory<nemo.core.neural_factory.NeuralModuleFactory>` and necessary :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>`
2) Defining a Directed Acyclic Graph (DAG) of :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>`
3) Call to "action" such as :class:`train<nemo.core.neural_factory.Actions.train>`

NeMo follows *lazy execution* model - actual computation happen only after training or inference is called.

:class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` is an abstraction between a layer and a neural network, for example: encoder, decoder, language model, acoustic model, etc. Each :class:`NeuralModule<nemo.core.neural_modules.NeuralModule>` computes a set of outputs from a set of inputs.
Every :class:`NmTensor<nemo.core.neural_types.NmTensor>` has :class:`NeuralType<nemo.core.neural_types.NeuralType>` .
:class:`NeuralType<nemo.core.neural_types.NeuralType>` describes tensor semantics, axis' order and dimensions. These types
are used to determine how modules should be connected together.

You can also pass to *callbacks* to actions which are then used for evaluation, logging and performance monitoring.





