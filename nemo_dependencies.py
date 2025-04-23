#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
NeMo dependency structure definition.
This module analyzes the codebase to determine internal dependencies between NeMo collections and core components.
"""

import ast
import json
import os
from typing import Dict, List, Set, Union


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    # Look in nemo directory and other relevant directories
    relevant_dirs = ['nemo', 'scripts', 'examples', 'tests']

    for dir_name in relevant_dirs:
        dir_path = os.path.join(directory, dir_name)
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

    return python_files


def analyze_imports(nemo_root: str, file_path: str) -> Set[str]:
    """Analyze a Python file and return its NeMo package dependencies using AST parsing."""
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)

        # Walk through the AST to find import statements

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('nemo.'):
                # Split the module path
                parts = node.module.split('.')

                if len(parts) == 1:
                    continue

                if len(parts) >= 2:
                    module_type = parts[1]  # collections, core, utils, or automodel

                    if module_type == 'collections':
                        if len(parts) >= 3:
                            # Handle both collection-level imports and specific module imports
                            collection = parts[2]
                            imported_package = f"nemo.collections.{collection}"
                            imports.add(imported_package)

                    elif module_type in find_top_level_packages(nemo_root):
                        imported_package = f"nemo.{module_type}"
                        imports.add(imported_package)

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

    return imports


def find_top_level_packages(nemo_root: str) -> List[str]:
    """Find all top-level packages under nemo directory."""
    packages: List[str] = []
    nemo_dir = os.path.join(nemo_root, 'nemo')

    if not os.path.exists(nemo_dir):
        print(f"Warning: nemo directory not found at {nemo_dir}")
        return packages

    for item in os.listdir(nemo_dir):
        item_path = os.path.join(nemo_dir, item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            packages.append(item)

    return sorted(packages)


def find_collection_modules(nemo_root: str) -> Dict[str, List[str]]:
    """Find all modules within collections."""
    collection_modules: Dict[str, List[str]] = {}
    collections_dir = os.path.join(nemo_root, 'nemo', 'collections')

    if not os.path.exists(collections_dir):
        print(f"Warning: collections directory not found at {collections_dir}")
        return collection_modules

    for collection in os.listdir(collections_dir):
        collection_path = os.path.join(collections_dir, collection)
        if os.path.isdir(collection_path) and not collection.startswith('__'):
            collection_modules[f"nemo.collections.{collection}"] = []

    return collection_modules


def build_dependency_graph(nemo_root: str) -> Dict[str, List[str]]:
    """Build a dependency graph by analyzing all Python files."""
    # Find all top-level packages
    top_level_packages = find_top_level_packages(nemo_root)
    print(f"Found top-level packages: {top_level_packages}")

    dependencies: Dict[str, List[str]] = {f"nemo.{top_level_package}": [] for top_level_package in top_level_packages}

    # Merge collection modules with top-level packages
    collection_modules = find_collection_modules(nemo_root)
    dependencies.update(collection_modules)

    # Second pass: analyze imports and build reverse dependencies
    for file_path in find_python_files(nemo_root):
        relative_path = os.path.relpath(file_path, nemo_root)
        parts = relative_path.split(os.sep)

        if len(parts) == 1 or parts[-1] == "__init__.py" or parts[0] != "nemo":
            continue

        if len(parts) == 2 and parts[1] in top_level_packages:
            dependencies[f"nemo.{parts[1]}"].extend(set(analyze_imports(nemo_root, file_path)))
            dependencies[f"nemo.{parts[1]}"] = list(set(dependencies[f"nemo.{parts[1]}"]))

        elif len(parts) >= 3 and parts[1] == 'collections':
            dependencies[f"nemo.collections.{parts[2]}"].extend(analyze_imports(nemo_root, file_path))
            dependencies[f"nemo.collections.{parts[2]}"] = list(set(dependencies[f"nemo.collections.{parts[2]}"]))
    return dependencies


def main():
    """Main function to analyze dependencies and output JSON."""
    # Get the root directory of the NeMo project
    nemo_root = os.path.dirname(os.path.abspath(__file__))

    # Build dependency graph
    dependencies = build_dependency_graph(nemo_root)

    # Output as JSON
    print(json.dumps(dependencies, indent=4))


if __name__ == "__main__":
    main()
