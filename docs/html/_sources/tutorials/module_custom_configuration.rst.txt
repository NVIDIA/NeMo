Customizing the configuration
-----------------------------


A generic configuration export enables to use of parameters of primitive types (string, int, float) \
or nested lists of/dicts of primitive types.

In order to extend that functionality by other, custom types one must overload the \
generic :meth:`export_to_config()` and  :meth:`import_from_config()` methods for his/her Module class. \
This tutorial explains how one can do it.


In the following example we will derive a class from the :class:`TaylorNet` (used in the previous example) \
and extend it by those methods. But first, let us define a simple :class:`Status` enum:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 33-35

Now let us define the :class:`CustomTaylorNet` Neural Module class:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 38-43


In order to properly handle the export of the :class:`Status` enum we must implement a custom function \
:meth:`export_to_config()`:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 45-76


Note that the configuration is actually a dictionary consisting of two sections:

 * ``header`` (storing class specification, NeMo version, NeMo collection name etc.) and
 * ``init_params`` storing the parameters used for instantiation of the object.

Those parameters are stored in the protected ``self._init_params``  field of the base :class:`NeuralModule` class.
It is assumed that (aside of this use-case) the user won't access nor use them directly.

Analogically, we must overload the :meth:`import_from_config()` method:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 79-119

Please note that the base :class:`NeuralModule` class provides several protected methods that we used, \
with most important being:

 * :meth:`_create_config_header()` generating the appropriate header, and \
 * :meth:`_validate_config_file()` validating the loaded configuration file (checking the header content).


.. note::
    It is once again worth emphasizing that the :meth:`import_from_config()` is a class method, actually returning a \
    new object instance - in this case of the hardcoded :class:`CustomTaylorNet` type.


Now we can simply create an instance and export its configuration by calling:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 128-129,134-135

And instantiate a second by loading that configuration:

.. literalinclude:: ../../../../examples/start_here/module_custom_configuration.py
   :language: python
   :lines: 137-139

As a result we will see that the new object has set the status to the same value as the original one:

.. code-block:: bash

    [NeMo I 2020-02-18 20:15:50 module_custom_configuration:74] Configuration of module 3ec99d30-baba-4e4c-a62b-e91268762864 (CustomTaylorNet) exported to /tmp/custom_taylor_net.yml
    [NeMo I 2020-02-18 20:15:50 module_custom_configuration:41] Status: Status.error
    [NeMo I 2020-02-18 20:15:50 module_custom_configuration:114] Instantiated a new Neural Module of type `CustomTaylorNet` using configuration loaded from the `/tmp/custom_taylor_net.yml` file
