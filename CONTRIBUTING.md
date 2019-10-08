# Contributions are welcome!

1) Please take a look at the LICENSE (it's Apache 2.0)

2) Make sure you sign your commits. E.g. use ``git commit -s`` when commiting

3) Make sure all unittests finish successfully before sending PR

4) Send your Pull Request to `master` branch


# Collection Guidelines
Collection is a logical grouping of related Neural Modules. It is a grouping of modules that share a domain area or semantics. At the basic level, collection is a python package installable via pip.
When contributing module to a collection, please make sure it belongs to that category. If you would like to start a new one and contribute back to the platform, you are very welcome to do so. Collection package should be named ``nemo_<collection-name>``. Collections can depend on other collections and have new types defined. Neural Types for input and output need to be clearly defined in documentation. 

Please note that CI needs to pass for all the modules and collections.

# Style guide

## General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
1. **Framework-agnostic**: separate backend code from NeMo code. For example, if something is based on PyTorch, it'd be wrapped around by a Neural Module before users can call it.
1. **Robust**: make it hard for users to make mistakes.
1. **Supporting of both training and inferencing**: if a module can only be used for training, write a companion module for inferencing.
1. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
1. **Readable**: code should be easier to read.
1. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo supports. Give credit and link back to the code.
1. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Python style
1. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
1. Minimize the use of ``**kwargs``.
1. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```.
1. Classes are preferred to standalone methods.
1. Methods should be atomic. A method shouldn't be longer than 75 lines, e.g. can be fit into the computer screen without scrolling.
1. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
1. Separate imports by built-in packages, installed packages, and internal packages. Arrange them by alphabetical orders.
1. Include docstrings for every class and method.
1. Add ``__init__.py`` for every folder.
1. F-strings are prefered to formatted strings.
1. Loggers are preferred to print. In NeMo, you can use logger from ``utils/exp_logging.py``
1. Private functions (functions start with ``_``) shouldn't be called outside its host file.
1. If a comment lasts multiple lines, use ``'''`` instead of ``#``.

## Nemo style
1. If you import a module from the same collection, use relative path instead of absolute path. For example, inside ``nemo_nlp``, use ``.utils`` instead of ``nemo_nelp.utils``.
1. Before accessing something, always make sure that it exists. E.g. right now, in ``actions.py``, there's this line of code ``batch_size=dl_nm.local_parameters["batch_size"]`` but nowhere in the codebase we check that ``batch_size`` is passed into datalayer.
1. Right inheritance. For example, if a module doesn't have any trainable weights, don't inherit from TrainableNM.
1. Naming consistency, both within NeMo and between NeMo and external literature. E.g. use the name ``logits`` for ``log_probs``, ``hidden_size`` for ``d_model``.
