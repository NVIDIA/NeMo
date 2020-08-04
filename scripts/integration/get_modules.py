# -*- coding: utf-8 -*-
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Script responsible for generation of a JSON file containing list of modules of a given collection. 

Args:
    Name of the collection (--c, --collection)

Returns:
    Format of the output JSON file (indicated  as --output_filename):

[
    {
        "name": "CIFAR100DataLayer",
        "id": "nemo.collections.cv.modules.data_layers.cifar100_datalayer.CIFAR100DataLayer",
        "module_type": "datalayer",
        "arguments": [
            {
                "name": "height",
                "default": 32,
                "annotation": "int"
            },
            {
                "name": "width",
                "default": 32,
                "annotation": "int"
            },
    ...
    },
    {
        "name": "CIFAR10DataLayer",
        "id": "nemo.collections.cv.modules.data_layers.cifar10_datalayer.CIFAR10DataLayer",
    ...
    },
...
]
"""


import argparse
import importlib
import inspect
import json
import os
import pkgutil

import nemo
from nemo.utils import logging


def process_member(module_name, obj, module_list) -> bool:
    """ Helper function processing the passed object and, if ok, adding a record to the module list.
    
    Args:
        name: name of the member
        obj: member (class/function etc.)
        module_list: list of modules that (probably) will be expanded.
    """

    # It is not a class - skip it.
    if not inspect.isclass(obj):
        return False

    # Skip abstract classes.
    if inspect.isabstract(obj):
        return False

    # Modules must inherit from datalayer/trainable/losses/nontraineble.
    parent_classes = {
        "datalayer": nemo.backends.pytorch.nm.DataLayerNM,
        "trainable": nemo.backends.pytorch.nm.TrainableNM,
        "loss": nemo.backends.pytorch.nm.LossNM,
        "nontrainable": nemo.backends.pytorch.nm.NonTrainableNM,
    }

    module_type = None
    # Check the inheritance.
    for type_name, cls in parent_classes.items():
        # Skip parent classes.
        if obj.__name__ == cls.__name__:
            return False
        # Check inheritance.
        if issubclass(obj, cls):
            module_type = type_name
            break
    if module_type is None:
        return False

    # Get ID.
    module_id = ".".join([obj.__module__, obj.__name__])
    logging.info("   * Processing `{}`".format(module_id))

    # Inspect the arguments.
    sig = inspect.signature(obj.__init__)

    # Invalid argument names.
    inv_params = ["*", "args", "kwargs"]
    module_args = []

    # Iterate over arguments.
    for name, param in sig.parameters.items():
        # Skip "self".
        if name == "self":
            continue
        # Skip invalid params.
        if name in inv_params:
            logging.warning("     ! Contains invalid argument `{}` - skipping the module".format(name))
            return False
        # Add name
        arg = {"name": name}

        # Get default.
        if param.default is not inspect.Signature.empty:
            # Check what is pased.
            if inspect.isclass(param.default):
                logging.warning(
                    "     ! Argument `{}` has invalid `default` set to `{}` - skipping the module".format(
                        name, param.default
                    )
                )
                return False
            else:
                arg["default"] = param.default

        # Get annotations.
        if param.annotation is not inspect.Signature.empty:
            # Check if the annotation is a "single type" or Union.
            if hasattr(param.annotation, "__name__"):
                arg["annotation"] = param.annotation.__name__
            else:
                # Handle Union.
                arg["annotation"] = str(param.annotation)

        # Add to list.
        module_args.append(arg)

    # Append "module" to list.
    module_list.append(
        {"name": module_name, "id": module_id, "module_type": module_type, "arguments": module_args,}
    )
    # Ok, got a module.
    return True


def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages

    Args:
        package: package (name or actual module) (str | module)

    Returns:
        dict[str, types.ModuleType]
    """
    # Check whether this is module or string describing the module that needs to be evaluated/imported.
    if isinstance(package, str):
        # Skip cache.
        if package == "__pycache__":
            return {}
        # Import module.
        package = importlib.import_module(package)
    results = {}
    # Walk through the subpackages.
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        # Try to import  the module.
        try:
            results[full_name] = importlib.import_module(full_name)
            # Dive into recursion.
            if recursive and is_pkg:
                results.update(import_submodules(full_name))
        except ModuleNotFoundError:
            # Simply skip it.
            continue

    return results


def get_modules():
    """ Main function analysing the indicated NeMo collection and generating a JSON file with module descriptions. """
    # Parse arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', '-c', help='ID of the collection', type=str, required=True)
    parser.add_argument(
        '--output_filename',
        '-o',
        help='Name of the output JSON file (DEFAULT: modules.json)',
        type=str,
        default="modules.json",
    )
    args = parser.parse_args()

    # Get collections directory.
    colletions_dir = os.path.dirname(nemo.collections.__file__)
    logging.info('Analysing collections in `{}`'.format(colletions_dir))

    # Generate list of NeMo collections - from the list of collection subfolders.
    collections = {}
    for sub_dir in os.listdir(colletions_dir):
        # Skip cache.
        if sub_dir == "__pycache__":
            continue
        # Check if it is a directory.
        if os.path.isdir(os.path.join(colletions_dir, sub_dir)):
            collections[sub_dir] = "nemo.collections." + sub_dir

    # Check the "user choice".
    if args.collection not in collections.keys():
        logging.error("Couldn't process the incidated `{}` collection".format(args.collection))
        logging.info(
            "Please select one of the existing collections using `--collection [{}]`".format("|".join(collections))
        )
        exit(-1)

    # Load the collection specification.
    collection_spec = importlib.util.find_spec(collections[args.collection])
    if collection_spec is None:
        logging.error("Failed to load the `{}` collection".format(val))
        exit(-2)

    # Iterate over the packages in the indicated collection.
    logging.info("Analysing the `{}` collection".format(args.collection))
    logging.info("=" * 80)

    # Import all submodules.
    submodules = import_submodules(collections[args.collection])

    # Extract list of neural modules.
    total_counter = 0
    module_list = []
    module_names = []
    # Iterate through submodules one by one.
    for sub_name, submodule in submodules.items():
        try:
            logging.info("* Analysing the `{}` module".format(sub_name))
            # Reset internal counters.
            counter = 0
            existing_counter = 0
            # Iterate throught members of a given submodule.
            for name, obj in inspect.getmembers(submodule):
                # Skip modules that might be already processed due to improper import scope
                # (enabling to access the same module by different name scopes).
                if name in module_names:
                    existing_counter += 1
                    continue
                # Process the member.
                if process_member(name, obj, module_list):
                    module_names.append(name)
                    counter += 1
            logging.info("  * Found {} already processed and {} new neural modules".format(existing_counter, counter))
        except AttributeError as e:
            logging.info("  * No neural modules were found")
        total_counter += counter

    # Add prefix - only for default name.
    output_filename = (
        args.output_filename
        if args.output_filename != "modules.json"
        else args.collection + "_" + args.output_filename
    )
    # Export to JSON.
    with open(output_filename, 'w') as outfile:
        json.dump(module_list, outfile)

    logging.info("=" * 80)
    modules = "".join("* {}\n".format(o) for o in module_list)
    logging.info(
        "Finished analysis of the `{}` collection, found {} modules: \n{}".format(
            args.collection, total_counter, modules
        )
    )
    logging.info('Results exported to `{}`.'.format(output_filename))


if __name__ == '__main__':
    get_modules()
