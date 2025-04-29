#!/usr/bin/env python3
import json
import os
import sys
from typing import Any, Dict, List, Set

import git

import nemo_dependencies


def get_changelog() -> List[Dict[str, Any]]:
    """
    Fetch the changelog between current branch and main.
    Returns a list of dictionaries containing commit information.
    """
    try:
        # Initialize the repo object - go up two levels from this file's location
        repo = git.Repo(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Get the current branch (PR branch)
        current = repo.active_branch

        # Get the main branch
        main = repo.heads.main

        # Get all commits between current branch and main
        commits = list(repo.iter_commits(f"{main.name}..{current.name}"))

        # Format the commits into a list of dictionaries
        changelog = []
        for commit in commits:
            # Get the list of changed files
            changed_files = []
            for diff in commit.diff(commit.parents[0] if commit.parents else None):
                changed_files.append(diff.a_path if diff.a_path else diff.b_path)

            changelog.append(
                {
                    'hash': commit.hexsha,
                    'message': commit.message.strip(),
                    'author': commit.author.name,
                    'date': commit.committed_datetime.isoformat(),
                    'files': changed_files,
                }
            )

        return changelog

    except git.exc.GitCommandError as e:
        print(f"Error fetching changelog: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def get_changed_files(changelog: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract a unique set of changed files from the changelog.

    Args:
        changelog: List of commit dictionaries from get_changelog()

    Returns:
        Set of unique file paths that were changed
    """
    changed_files = set()
    for commit in changelog:
        changed_files.update(commit['files'])
    return changed_files


def main():
    """
    Main function to fetch and output the changelog and changed files.
    """
    changelog = get_changelog()

    # Output the changelog as JSON
    print("Changelog:")
    print(json.dumps(changelog, indent=2))

    # Output unique changed files
    print("\nChanged files:")
    changed_files = get_changed_files(changelog)
    print(json.dumps(sorted(list(changed_files)), indent=2))

    nemo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Build dependency graph
    dependencies = nemo_dependencies.build_dependency_graph(nemo_root)

    test_modules = []
    for changed_file in changed_files:
        if changed_file in dependencies:
            test_modules.extend(dependencies[changed_file])

    test_modules = list(set(test_modules))

    print(test_modules)


if __name__ == "__main__":
    main()
