Module Configuration
====================

Neural Modules have configuration that can be imported from/exported to YAML file. \
A module configuration file stores all parameters required for creation of an instance.

.. note::
    In the case of Trainable Neural Modules the `configuration` is complementary to checkpoint, i.e. \
    configuration contains parameters (like e.g. number of layers, hidden size etc.), \
    whereas checkpoint contains the actual module weights.


Example
------------

In the following example we will once again train a model to learn Taylor's coefficients for y=sin(x). \
However, we will extend the example by showing how to export configuration of the module to a YAML file and \
create a second instance having the same set of parameters.

Let us start by creating the Neural Factory and instatiating the modules from the original example:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 20-33

Now we can export the configuration of any of the existing modules by using  the `export_to_config()`, for \
example let us export the configuration of the trainable `TaylorNet` like this:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 36

There is an analogical function `import_from_config()` responsible for loading the configuration file:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 39

.. note::
    The `import_from_config()` function actually creates a new instance of object of the class that was stored in the \
    configuration. But it is important to understand that both instances do not share any trainable weights. \
    NeMo offers a separate mechanism for weight tying.

Now we can use the newly imported module in the same way as every other module. \
For example, we can build a graph and train it with a NeMo trainer:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 41-



.. note::
    This and other examples can be found in the `nemo/examples` folder
