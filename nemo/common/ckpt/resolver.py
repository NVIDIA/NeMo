from functools import wraps
from typing import Callable, Dict, Optional, Union
from pathlib import Path
import os

import fsspec

CHECKPOINT_RESOLVERS: Dict[str, Callable[[Path], Optional[Path]]] = {}


def get_checkpoint(
    path: Optional[Union[str, Path]],
    resolver: Optional[Union[Callable[[Path], Optional[Path]], str]] = None,
) -> Optional[Path]:
    """Resolves checkpoint paths using an optional resolution strategy.

    Args:
        path: The input path to resolve
        resolver: Either a callable that takes a Path and returns a Path,
        or a string matching a registered resolver

    Returns:
        Resolved Path object or None if input path is None

    Example:
        # Using a registered resolver
        path = get_checkpoint("/path/to/checkpoints", resolver="latest")

        # Using a custom resolver
        def my_resolver(path: Path) -> Optional[Path]:
            return path / "best_checkpoint"
        path = get_checkpoint("/path/to/checkpoints", resolver=my_resolver)
    """
    if path is None:
        return None

    # Use fsspec to handle the path
    fs, path = fsspec.core.url_to_fs(str(path))

    # Resolve the resolver callable
    resolver_fn = None
    if resolver is not None:
        if isinstance(resolver, str):
            if resolver not in CHECKPOINT_RESOLVERS:
                raise ValueError(
                    f"Unknown resolver strategy '{resolver}'. "
                    f"Available strategies: {list(CHECKPOINT_RESOLVERS.keys())}"
                )
            resolver_fn = CHECKPOINT_RESOLVERS[resolver]
        else:
            resolver_fn = resolver

    # Apply resolution if provided
    if resolver_fn is not None:
        resolved_path = resolver_fn(path)
        if resolved_path is not None:
            path = resolved_path

    return Path(path)


def register_checkpoint_resolver(name: str):
    """Decorator to register checkpoint resolution strategies.

    Args:
        name: Name of the resolution strategy

    Example:
        @register_checkpoint_resolver("my_strategy")
        def my_resolver(path: Path) -> Optional[Path]:
            # Custom resolution logic
            return resolved_path
    """

    def decorator(
        func: Callable[[Path], Optional[Path]],
    ) -> Callable[[Path], Optional[Path]]:
        @wraps(func)
        def wrapper(path: Path) -> Optional[Path]:
            return func(path)

        CHECKPOINT_RESOLVERS[name] = wrapper
        return wrapper

    return decorator


@register_checkpoint_resolver("latest")
def latest_checkpoint_resolver(path: Path) -> Optional[Path]:
    """Resolves to the latest checkpoint in a directory using NeMo's existing logic.

    Args:
        path: Directory path containing checkpoints
    Returns:
        Path to the latest valid checkpoint or None
    """
    fs, path = fsspec.core.url_to_fs(str(path))

    if not fs.exists(path):
        return None

    checkpoints_path = f"{path}/checkpoints"
    if fs.exists(checkpoints_path):
        path = checkpoints_path

    # List all files/directories in the path
    try:
        all_files = fs.ls(path, detail=True)
    except Exception:
        return None

    # Filter for directories
    dist_checkpoints = [f for f in all_files if f["type"] == "directory"]

    # Helper to check if path matches pattern
    def matches_pattern(file_info, pattern):
        file_name = os.path.basename(file_info["name"])
        return any(p in file_name for p in pattern)

    # Filter for end and last checkpoints
    end_checkpoints = [
        f
        for f in dist_checkpoints
        if matches_pattern(f, ["end"]) and not matches_pattern(f, [".unfinished"])
    ]
    last_checkpoints = [
        f
        for f in dist_checkpoints
        if matches_pattern(f, ["last"]) and not matches_pattern(f, [".unfinished"])
    ]

    if end_checkpoints:
        if len(end_checkpoints) > 1:
            if "mp_rank" in str(end_checkpoints[0]["name"]):
                return end_checkpoints[0]["name"]
            raise ValueError(
                f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt."
            )
        return end_checkpoints[0]["name"]

    if len(last_checkpoints) > 1:
        if any(
            [
                s
                for s in ["mp_rank", "tp_rank", "fsdp_shard"]
                if s in str(last_checkpoints[0]["name"])
            ]
        ):
            checkpoint = last_checkpoints[0]["name"]
            # Note: uninject_model_parallel_rank needs to be adapted for fsspec paths
            return checkpoint
        # Select checkpoint with latest modified time
        return sorted(last_checkpoints, key=lambda x: x["mtime"], reverse=True)[0][
            "name"
        ]

    return last_checkpoints[0]["name"] if last_checkpoints else None
