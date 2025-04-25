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
                            # imports.add(node.module)
                            if node.names:
                                for name in node.names:
                                    if name.name == '*':
                                        continue

                                    imports.add(f"{node.module}.{name.name}")

                    elif module_type in find_top_level_packages(nemo_root):
                        imported_package = f"nemo.{module_type}"
                        # imports.add(node.module)
                        if node.names:
                            for name in node.names:
                                if name.name == '*':
                                    continue
                                imports.add(f"{node.module}.{name.name}")

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

    dependencies: Dict[str, List[str]] = {}

    # Second pass: analyze imports and build reverse dependencies
    for file_path in find_python_files(nemo_root):
        relative_path = os.path.relpath(file_path, nemo_root)
        parts = relative_path.split(os.sep)

        if len(parts) == 1 or parts[-1] == "__init__.py" or parts[0] != "nemo":
            continue

        module_path = relative_path.replace(".py", "").replace("/", ".")
        if parts[1] in top_level_packages and parts[1] != 'collections':
            dependencies[module_path] = list(set(analyze_imports(nemo_root, file_path)))
        elif parts[1] == 'collections':
            dependencies[module_path] = list(set(analyze_imports(nemo_root, file_path)))

    # Flip the dependency graph to show reverse dependencies
    reverse_dependencies: Dict[str, List[str]] = {}
    # Handle top-level package dependencies
    for package, deps in dependencies.items():
        for dep in deps:
            if dep not in reverse_dependencies:
                reverse_dependencies[dep] = []
            reverse_dependencies[dep].append(package)
    dependencies = reverse_dependencies

    # Simplify values: Either top-level package or collection module
    simplified_dependencies: Dict[str, List[str]] = {}
    for package, deps in dependencies.items():
        simplified_deps = []
        for dep in deps:
            dep_parts = dep.split('.')

            if package not in simplified_dependencies:
                simplified_dependencies[package] = []

            if len(parts) == 2 and (simplified_name := f"{dep_parts[0]}.{dep_parts[1]}") in find_top_level_packages(
                nemo_root
            ):
                simplified_dependencies[package].append(simplified_name)

            elif len(parts) >= 3 and (
                simplified_name := f"{dep_parts[0]}.{dep_parts[1]}.{dep_parts[2]}"
            ) in find_collection_modules(nemo_root):
                simplified_dependencies[package].append(simplified_name)

            simplified_dependencies[package] = list(set(simplified_dependencies[package]))

    # # Clean up package names to match file paths
    # cleaned_dependencies = {}
    # for package, deps in dependencies.items():
    #     # Convert package path to filesystem path for checking
    #     parent_package = ".".join(package.split(".")[:-1])
    #     parent_package_path = os.path.join(f"{parent_package.replace('.', '/')}/__init__.py")
    #     parent_module_path = os.path.join(f"{parent_package.replace('.', '/')}.py")

    #     if parent_package == "nemo.collections.common.tokenizers":
    #         print(parent_package_path)
    #         print(parent_module_path)
    #         print(parent_package)

    #     if os.path.isfile(parent_package_path):
    #         if parent_package_path in cleaned_dependencies:
    #             cleaned_dependencies[parent_package_path].extend(deps)
    #         else:
    #             cleaned_dependencies[parent_package_path] = deps

    #     elif os.path.isfile(parent_module_path):
    #         if parent_module_path in cleaned_dependencies:
    #             cleaned_dependencies[parent_module_path].extend(deps)
    #         else:
    #             cleaned_dependencies[parent_module_path] = deps
    #     else:
    #         if package in cleaned_dependencies:
    #             cleaned_dependencies[package].extend(deps)
    #         else:
    #             cleaned_dependencies[package] = deps

    # for package, deps in cleaned_dependencies.items():
    #     cleaned_dependencies[package] = list(set(cleaned_dependencies[package]))

    # dependencies = cleaned_dependencies

    # # Follow and extend records with transitive dependencies
    # transitive_dependencies = dependencies.copy()

    # # Keep iterating until no new dependencies are added
    # while True:
    #     changes_made = False
    #     new_dependencies = transitive_dependencies.copy()

    #     # For each package and its direct dependencies
    #     for package, deps in transitive_dependencies.items():
    #         # For each direct dependency
    #         for dep in deps:
    #             # If the dependency has its own dependencies
    #             if dep in transitive_dependencies:
    #                 # Add those transitive dependencies to the original package
    #                 for transitive_dep in transitive_dependencies[dep]:
    #                     if transitive_dep not in new_dependencies[package]:
    #                         new_dependencies[package].append(transitive_dep)
    #                         changes_made = True

    #     # Update dependencies with new transitive ones
    #     transitive_dependencies = new_dependencies

    #     # If no new dependencies were added, we're done
    #     if not changes_made:
    #         break

    # dependencies = transitive_dependencies

    # # Sort dependencies by length of values (number of dependencies)
    # dependencies = dict(sorted(dependencies.items(), key=lambda x: len(x[1]), reverse=True))

    return dependencies


def main():
    """Main function to analyze dependencies and output JSON."""
    # Get the root directory of the NeMo project
    nemo_root = os.path.dirname(os.path.abspath(__file__))

    # Build dependency graph
    dependencies = build_dependency_graph(nemo_root)

    # Output as JSON
    data = json.dumps(dependencies, indent=4)
    # print(data)
    with open('nemo_dependencies.json', 'w') as f:
        f.write(data)


if __name__ == "__main__":
    main()
