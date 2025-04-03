# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import subprocess

VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0


def get_git_commit_sha_short():
    """
    Returns the short SHA-1 hash of the current commit, or 'unknown' if not in a git repository.
    """
    try:
        # Run git command to get the short commit hash (7 characters)
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_sha = "unknown"
    return commit_sha


def is_git_tree_dirty():
    """
    Returns whether the git tree is dirty (has uncommitted changes).
    """
    try:
        # Check for uncommitted changes
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL)
        return dirty != 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_version():
    """
    Returns the current version of the module in the format:
    VERSION_MAJOR.VERSION_MINOR.VERSION_PATCH-GIT_COMMIT_SHA_SHORT[+GIT_TREE_DIRTY]
    """
    # Get the short commit hash
    git_commit_sha = get_git_commit_sha_short()

    # Check if the git tree is dirty
    git_tree_dirty = is_git_tree_dirty()

    # Construct the version string
    version = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}+{git_commit_sha}"

    # Append "+dirty" if the git tree is not clean
    if git_tree_dirty:
        version += ".dirty"

    return version


# Example usage
if __name__ == "__main__":
    print(get_version())
