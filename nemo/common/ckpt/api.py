from pathlib import Path
import shutil


UNFINISHED_CHECKPOINT_SUFFIX: str = ".unfinished"


def is_unfinished(path: Path) -> bool:
    """Check if a checkpoint path has an unfinished marker."""
    marker = path.parent / f"{path.name}{UNFINISHED_CHECKPOINT_SUFFIX}"
    return marker.exists()


def cleanup_unfinished(directory: Path) -> None:
    """Remove any unfinished checkpoints in directory.

    Args:
        directory: Directory to clean up
    """
    # Find all unfinished markers
    for marker in directory.glob(f"*{UNFINISHED_CHECKPOINT_SUFFIX}"):
        ckpt_path = Path(str(marker)[: -len(UNFINISHED_CHECKPOINT_SUFFIX)])
        if ckpt_path.exists():
            print(f"Removing unfinished checkpoint: {ckpt_path}")
            shutil.rmtree(ckpt_path)
        marker.unlink()


def find_latest(directory: Path, exclude_unfinished: bool = True, pattern: str = None) -> Path | None:
    """Find the latest checkpoint in a directory.

    Args:
        directory: Directory to search
        exclude_unfinished: Whether to exclude checkpoints marked as unfinished
        pattern: Optional glob pattern to filter checkpoints (e.g., "*last" or "*end")
    """
    if not directory.exists():
        return None
        
    checkpoints = []
    # Use pattern if provided, otherwise get all items
    glob_pattern = pattern if pattern else "*"
    
    for path in directory.glob(glob_pattern):
        if exclude_unfinished and is_unfinished(path):
            continue
        checkpoints.append(path)

    if not checkpoints:
        return None

    # Handle the case with multiple checkpoints
    if len(checkpoints) > 1:
        # Check if these are distributed training checkpoints
        if any(["mp_rank" in str(p) or "tp_rank" in str(p) or "fsdp_shard" in str(p) for p in checkpoints]):
            # For distributed checkpoints, just take the first one
            # In a real implementation, you might want to handle this more carefully
            return Path(checkpoints[0])
        else:
            # Sort by modification time and take the most recent
            latest = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            return Path(latest)
    
    # Single checkpoint case
    return Path(checkpoints[0])
