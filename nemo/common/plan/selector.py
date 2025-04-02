from torch import nn
import fnmatch
import weakref
import logging
import dataclasses
import collections.abc
import inspect
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Optional,
    Iterator,
    Union,
    Callable,
    Type,
    Set,
)

from nemo.common.plan.plan import Plan


StrIteratorFunc = Callable[[Any], Iterator[Tuple[str, Any]]]
PredicateFunc = Callable[[Any], bool]
IteratorRegistryKey = Union[Type, PredicateFunc]


@dataclasses.dataclass
class MatchContext:
    """Context provided to a Plan during its execution.

    Attributes:
        path: The full path string to the item the plan is being applied to.
        root: The root object on which the Selector was materialized.
        module: The specific item instance the plan is being applied to.
                Renamed from 'module' to 'target_item' for clarity as it
                might not always be an nn.Module.
    """

    path: str = ""
    root: Optional[Any] = None
    target_item: Optional[Any] = None  # Renamed from 'module' for generality


class MaterializedSelector:
    """Represents the resolved state of a Selector for a specific object.

    Holds the computed mapping from matched paths to Plans and applies them
    during execution. This object is typically created by `Selector.materialize()`.
    """

    def __init__(self, path_to_plan: Dict[str, "Plan"], target_obj: Any):
        """Initializes the MaterializedSelector.

        Args:
            path_to_plan: The dictionary mapping resolved paths to Plan instances.
            target_obj: The original root object this selector was materialized for
                        (stored as a weak reference).
        """
        if not isinstance(path_to_plan, dict):
            raise TypeError(f"path_to_plan must be a dict, got {type(path_to_plan)}")
        self._path_to_plan = path_to_plan
        self._target_ref = weakref.ref(target_obj)
        logging.debug(
            f"Created MaterializedSelector with {len(path_to_plan)} matched paths."
        )

    def items(self) -> Iterator[Tuple[str, "Plan"]]:
        """Returns an iterator over the (path, Plan) pairs."""
        return iter(self._path_to_plan.items())

    def execute(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
        """Executes the materialized plans on the target object structure.

        Applies the associated Plan to each item found at the pre-computed paths.
        Modifications happen in-place on the provided `obj`.

        Args:
            obj: The object structure to apply plans to. Should ideally be the
                 same object used for materialization.
            *args: Positional arguments passed through to each Plan's execution.
            **kwargs: Keyword arguments passed through to each Plan's execution
                      (potentially merged with MatchContext if the plan accepts it).

        Returns:
            The potentially modified object `obj`.
        """
        target = self._target_ref()
        if target is not None and id(obj) != id(target):
            logging.warning(
                "Executing MaterializedSelector on a different object than it was materialized for. "
                "Path resolution might be incorrect if the structure differs."
            )
        elif target is None:
            logging.warning(
                "Executing MaterializedSelector after the original target object was garbage collected. "
                "Results may be unpredictable."
            )

        # Apply plans in sorted path order. This often helps ensure parent nodes
        # are processed before children if patterns target both levels, although
        # complex modifications might still have order dependencies.
        logging.debug(f"Executing {len(self._path_to_plan)} plans...")
        executed_count = 0
        skipped_count = 0
        for path in sorted(self._path_to_plan.keys()):
            plan = self._path_to_plan[path]
            result = self._apply_plan_to_path(obj, path, plan, *args, **kwargs)
            if result is not None:  # _apply_plan_to_path returns None on failure
                executed_count += 1
            else:
                skipped_count += 1

        logging.debug(
            f"Plan execution finished. Executed: {executed_count}, Skipped/Failed: {skipped_count}"
        )
        return obj  # Return the potentially modified object

    def _apply_plan_to_path(
        self, root_obj: Any, path: str, plan: "Plan", *args: Any, **kwargs: Any
    ) -> Optional[Any]:
        """Locates the item at `path` within `root_obj` and applies the `plan`.

        Handles nested attribute access ('.'), list/tuple indexing ('[int]'),
        and dictionary key access ('[repr(key)]'). Updates the parent container
        if the plan returns a non-None value (for mutable containers).

        Args:
            root_obj: The top-level object to navigate.
            path: The path string specifying the target item.
            plan: The Plan instance to apply.
            *args: Positional arguments for the plan.
            **kwargs: Keyword arguments for the plan.

        Returns:
            The result returned by the plan's execution, or None if path traversal
            or plan application failed.
        """
        parent = None
        target_obj = None
        # Stores the actual attribute name, index, or key used for the final access
        accessor: Union[str, int] = "<root>"

        try:
            # 1. Locate Target Object using dot-separated path
            if path == "":
                parent = None
                target_obj = root_obj
            else:
                # Split path: remove leading dot if present, then split by '.'
                segments = path.lstrip(".").split(".")
                current = root_obj

                for i, segment in enumerate(segments):
                    if not segment:  # Skip empty segments if double dots occur '..'
                        continue
                    is_last = i == len(segments) - 1
                    parent = current  # Store parent before access attempt

                    next_obj = None
                    segment_accessor: Union[str, int] = segment  # Default to string

                    # --- Access attempt with precedence ---
                    # 1. Try Attribute Access
                    try:
                        next_obj = getattr(current, segment)
                        segment_accessor = segment  # Accessor is the attribute name string
                        accessed = True
                    except AttributeError:
                        accessed = False

                    # 2. Try Integer Index Access (if attribute failed)
                    if not accessed:
                        try:
                            idx = int(segment)
                            next_obj = current[idx]
                            segment_accessor = idx  # Accessor is the integer index
                            accessed = True
                        except (ValueError, TypeError, IndexError):  # Not int / not sequence / out of bounds
                            accessed = False

                    # 3. Try String Key Access (if attribute and index failed)
                    if not accessed:
                        try:
                            # Use the original segment string as the key
                            next_obj = current[segment]
                            segment_accessor = segment  # Accessor is the string key
                            accessed = True
                        except (TypeError, KeyError, IndexError):  # Not dict / key missing / not sequence
                            # If all access methods fail
                            raise ValueError(
                                f"Cannot resolve segment '{segment}' in path '{path}' on object type {type(current).__name__}"
                            ) from None

                    # --- Update for next iteration or final target ---
                    if is_last:
                        target_obj = next_obj
                        accessor = segment_accessor  # Store the successful final accessor
                        break
                    else:
                        current = next_obj  # Move to the next object in the path

            if target_obj is None and path != "":
                raise ValueError(f"Path traversal failed for path: '{path}'")

            # 2. Apply Plan
            context = MatchContext(path=path, root=root_obj, target_item=target_obj)
            plan_callable = getattr(plan, "execute", None) or getattr(plan, "__call__", None)
            pass_context = False
            if plan_callable:
                try:
                    sig = inspect.signature(plan_callable)
                    if "context" in sig.parameters:
                        pass_context = True
                except (ValueError, TypeError):
                    pass  # Ignore errors inspecting signature

            apply_kwargs = kwargs.copy()
            if pass_context:
                apply_kwargs["context"] = context

            logging.debug(
                f"Applying plan {plan!r} to path '{path}' (target type: {type(target_obj).__name__})"
            )
            transformed = plan(target_obj, *args, **apply_kwargs)

            # 3. Update Parent (if transformed and possible)
            if transformed is not None and parent is not None and path != "":
                logging.debug(
                    f"Plan returned non-None value, attempting update on parent {type(parent).__name__} using {accessor!r}"
                )
                is_set = False
                try:
                    # Use the determined accessor (attribute name, index, or key)
                    if isinstance(accessor, int) and isinstance(
                        parent, list
                    ):  # Only list is mutable by index
                        if 0 <= accessor < len(parent):
                            parent[accessor] = transformed
                            is_set = True
                        else:
                            logging.error(
                                f"Index {accessor} OOB [{len(parent)}] for list @ {path}"
                            )
                    elif isinstance(
                        parent, dict
                    ):  # Assumes accessor could be any hashable key used by dict
                        parent[accessor] = transformed
                        is_set = True
                    elif isinstance(accessor, str):  # Assume attribute
                        setattr(parent, accessor, transformed)
                        is_set = True
                    else:
                        # Includes tuple index case (immutable) and other unhandled accessor types
                        logging.warning(
                            f"Cannot (or should not) set value for accessor '{accessor}' ({type(accessor)}) on parent type {type(parent).__name__} @ {path}. Modification skipped."
                        )

                    if is_set:
                        logging.debug(f"Successfully updated parent at path '{path}'")

                except Exception as e_set:
                    logging.error(
                        f"Failed to set transformed value at path '{path}' on parent {type(parent)}: {e_set}"
                    )

            return transformed  # Return result of plan execution

        except Exception as e:
            # Log detailed error if path traversal or application fails
            target_type_str = (
                type(target_obj).__name__
                if "target_obj" in locals() and target_obj is not None
                else "N/A"
            )
            parent_type_str = (
                type(parent).__name__
                if "parent" in locals() and parent is not None
                else "N/A"
            )
            logging.warning(
                f"Could not apply plan to path '{path}'. "
                f"Root: {type(root_obj).__name__}, Target Type: {target_type_str}, Parent Type: {parent_type_str}. "
                f"Error: {e}",
                exc_info=True,  # Log full traceback for warnings
            )
            return None  # Indicate failure

    def materialize(self, obj: Any) -> "MaterializedSelector":
        """Materialized selectors are already materialized. Returns self."""
        # Could add check: if id(obj) != id(self._target_ref()): warn?
        return self

    def __repr__(self) -> str:
        """Returns a dictionary-style representation of the path-to-plan map."""
        class_name = self.__class__.__name__
        if not self._path_to_plan:
            return f"{class_name}({{}})"
        items_str = "\n"
        for path in sorted(self._path_to_plan.keys()):  # Sort for consistent output
            plan = self._path_to_plan[path]
            # Use repr for path for correct quote handling, special case root
            path_repr = repr(path) if path else "'<root>'"
            items_str += f"  {path_repr}: {plan!r},\\n"  # Use !r for plan repr
        return f"{class_name}({{{items_str}}})"


class Selector:
    """Selects items in a data structure based on path patterns using registered iterators.

    Uses `fnmatch` for pattern matching against generated paths during traversal.
    Traverses common types like `nn.Module`, dataclasses, lists, tuples, and dicts
    by default using registered iterators. Custom container types can be supported
    by registering a suitable iterator function using `@Selector.register_iterator`.

    The `materialize()` method resolves patterns for a given object structure,
    returning an efficient `MaterializedSelector` for execution. Selector instances
    can also be called directly, which implicitly calls `materialize()` and `execute()`.
    """

    _iterators: Dict[IteratorRegistryKey, StrIteratorFunc] = {}
    """Registry mapping types or predicates to functions yielding (str_key_or_index, child)."""

    def __init__(self, patterns: Dict[str, "Plan"]):
        """Initializes the Selector.

        Args:
            patterns: A dictionary mapping `fnmatch` pattern strings to `Plan`
                      instances. Later patterns in the dictionary override earlier
                      ones if they match the same path.
        """
        if not isinstance(patterns, dict):
            raise TypeError(f"patterns must be a dict, got {type(patterns)}")
        self._patterns = patterns
        self._materialized_cache = weakref.WeakKeyDictionary()
        # Default iterators are registered via decorators when the class is defined.

    @classmethod
    def register_iterator(
        cls, target_key: IteratorRegistryKey
    ) -> Callable[[StrIteratorFunc], StrIteratorFunc]:
        """Decorator registers func to iterate children of `target_type` or matching `predicate`.

        Func yields `(str_key_or_index, child_item)` tuples.
        """
        def decorator(func: StrIteratorFunc) -> StrIteratorFunc:
            if target_key in cls._iterators:
                key_repr = target_key.__name__ if hasattr(target_key, '__name__') else repr(target_key)
                logging.debug(f"Overwriting existing iterator for key {key_repr}")
            cls._iterators[target_key] = func
            key_repr = target_key.__name__ if hasattr(target_key, '__name__') else repr(target_key)
            logging.debug(f"Registered iterator for key {key_repr}")
            return func
        return decorator

    def _get_iterator(self, obj: Any) -> Optional[StrIteratorFunc]:
        """Find the most specific registered iterator using predicates first, then MRO types."""
        # 1. Check Predicates
        # Iterate through items to check predicate keys
        for key, iterator_func in self._iterators.items():
            # Check if key is a callable but not a type (i.e., a predicate)
            if callable(key) and not isinstance(key, type):
                try:
                    if key(obj): # Call the predicate
                        return iterator_func
                except Exception as e_pred:
                    # Log errors during predicate evaluation but continue searching
                    logging.debug(f"Error calling predicate {key.__name__ if hasattr(key, '__name__') else key} on {type(obj)}: {e_pred}")
                    continue # Don't stop if one predicate fails

        # 2. Check Type Hierarchy (MRO)
        for obj_cls in type(obj).__mro__:
            # Check if the class itself is registered as a key
            if isinstance(obj_cls, type) and obj_cls in self._iterators:
                # Ensure the registered key *is* the type we're checking
                registered_key = obj_cls
                # Double-check we are fetching the function associated with the type key
                if registered_key in self._iterators:
                    # Confirm the key is a type (not a predicate that happens to equal the class)
                    if isinstance(registered_key, type):
                        return self._iterators[registered_key]

        return None # No suitable iterator found

    def materialize(self, obj: Any) -> "MaterializedSelector":
        """Resolves patterns against the object structure and returns a MaterializedSelector.

        Traverses the object using registered iterators, generates paths, matches
        patterns using `fnmatch`, and builds the path-to-plan map. Caches the
        result if the root object `obj` is hashable.

        Args:
            obj: The root object structure to analyze.

        Returns:
            A `MaterializedSelector` instance ready for execution.
        """
        is_hashable = isinstance(obj, collections.abc.Hashable)

        # 1. Check Cache
        if is_hashable:
            cached_result = self._materialized_cache.get(obj)
            if cached_result:
                logging.debug(
                    f"Using cached MaterializedSelector for object id {id(obj)}"
                )
                return cached_result

        # 2. Perform Traversal
        logging.debug(f"Materializing Selector for object id {id(obj)}")
        path_to_plan: Dict[str, Plan] = {}
        visited_ids: Set[int] = set()  # For cycle detection

        def _traverse(current_obj: Any, current_path: str):
            """Recursive helper to traverse structure and match patterns."""
            # --- Pre-traversal checks ---
            try:
                obj_id = id(current_obj)
            except TypeError:  # Handle unhashable objects (treat as leaves)
                obj_type_name = getattr(type(current_obj), "__name__", "<unhashable>")
                logging.debug(
                    f"Cannot get ID for object {obj_type_name} at path '{current_path}', treating as leaf."
                )
                # Still attempt pattern match on this node before returning
                for pattern, plan in self._patterns.items():
                    if fnmatch.fnmatch(current_path, pattern) or fnmatch.fnmatch(
                        obj_type_name, pattern
                    ):
                        path_to_plan[current_path] = plan
                return

            if obj_id in visited_ids:
                return  # Cycle detected

            # --- Leaf/Simple type check (stop recursion) ---
            # Check common immutable types and None. Also stop if no iterator found later.
            if current_obj is None or isinstance(
                current_obj, (str, int, float, bool, bytes, type)
            ):
                obj_type_name = getattr(type(current_obj), "__name__", "<unknown_type>")
                # Match leaf node path/type before returning
                for pattern, plan in self._patterns.items():
                    if fnmatch.fnmatch(current_path, pattern) or fnmatch.fnmatch(
                        obj_type_name, pattern
                    ):
                        path_to_plan[current_path] = plan
                return

            # --- Mark as visited ---
            visited_ids.add(obj_id)

            try:
                # --- Match patterns against the current node ---
                obj_type_name = getattr(type(current_obj), "__name__", "<unknown_type>")
                for pattern, plan in self._patterns.items():
                    # Check path and type name; last matching pattern wins
                    if fnmatch.fnmatch(current_path, pattern) or fnmatch.fnmatch(
                        obj_type_name, pattern
                    ):
                        path_to_plan[current_path] = plan

                # --- Find iterator and recurse ---
                iterator_func = self._get_iterator(current_obj)

                if iterator_func:
                    logging.debug(
                        f"Traversing children of '{current_path}' ({obj_type_name}) using {iterator_func.__name__}"
                    )
                    child_count = 0
                    for key_or_index_str, child_item in iterator_func(current_obj):
                        # Ensure key_or_index_str is a valid string for path construction
                        if not isinstance(
                            key_or_index_str, str
                        ) or not key_or_index_str:
                            logging.warning(
                                f"Iterator for {obj_type_name} yielded invalid segment '{key_or_index_str}' at path '{current_path}'. Skipping child."
                            )
                            continue
                        # Construct full path for child
                        child_path = f"{current_path}.{key_or_index_str}" if current_path else key_or_index_str
                        _traverse(child_item, child_path)
                        child_count += 1
                    logging.debug(
                        f"Finished traversing {child_count} children of '{current_path}'"
                    )

                # else: Treat as leaf (patterns were checked above)

            except Exception as e_traverse:
                # Log errors during node processing or child iteration
                logging.warning(
                    f"Error during traversal at/near path '{current_path}' for object {type(current_obj).__name__}: {e_traverse}",
                    exc_info=False,
                )
            finally:
                # Remove from visited *after* exploring children
                visited_ids.remove(obj_id)

        # --- Start Traversal ---
        _traverse(obj, "")  # Start with empty path for the root object

        # 3. Create and Cache Result
        materialized_selector = MaterializedSelector(path_to_plan, obj)
        if is_hashable:
            try:
                self._materialized_cache[obj] = materialized_selector
                logging.debug(f"Cached MaterializedSelector for object id {id(obj)}")
            except TypeError as e_cache:
                logging.warning(
                    f"Failed to cache MaterializedSelector for object id {id(obj)}: {e_cache}"
                )  # Should be rare

        logging.info(
            f"Selector materialization complete. Found {len(path_to_plan)} matching paths."
        )
        return materialized_selector

    # --- Convenience Methods ---

    def get_matches(self, obj: Any) -> Dict[str, "Plan"]:
        """Generates the path->plan map for the object without caching (or uses cache)."""
        materialized = self.materialize(obj)
        # Return a copy to prevent external modification of internal state
        return materialized._path_to_plan.copy()

    def show_matches(self, obj: Any) -> str:
        """Returns a string showing which patterns matched which resolved paths."""
        # This re-runs traversal logic to accurately map paths back to the
        # patterns that selected them. Less efficient but simpler than storing
        # pattern info in the materialized map itself.
        materialized = self.materialize(obj)  # Ensures map is computed
        path_to_plan = materialized._path_to_plan
        pattern_to_paths: Dict[str, List[str]] = {p: [] for p in self._patterns.keys()}
        visited_ids: Set[int] = set()

        def _find_matches(current_obj: Any, current_path: str):
            """Helper to re-run traversal and check which patterns match."""
            try:
                obj_id = id(current_obj)
            except TypeError:  # Handle unhashable leaf nodes
                if current_path in path_to_plan:
                    obj_type_name = getattr(
                        type(current_obj), "__name__", "<unknown_type>"
                    )
                    for pattern, plan in self._patterns.items():
                        if (
                            plan is path_to_plan[current_path]
                        ):  # Check if plan instance matches
                            if (
                                fnmatch.fnmatch(current_path, pattern)
                                or fnmatch.fnmatch(obj_type_name, pattern)
                            ) and current_path not in pattern_to_paths[pattern]:
                                pattern_to_paths[pattern].append(current_path)
                return

            if (
                obj_id in visited_ids
                or current_obj is None
                or isinstance(current_obj, (str, int, float, bool, type))
            ):
                # Leaf node check
                if current_path in path_to_plan:
                    obj_type_name = getattr(
                        type(current_obj), "__name__", "<unknown_type>"
                    )
                    for pattern, plan in self._patterns.items():
                        if plan is path_to_plan[current_path]:
                            if (
                                fnmatch.fnmatch(current_path, pattern)
                                or fnmatch.fnmatch(obj_type_name, pattern)
                            ) and current_path not in pattern_to_paths[pattern]:
                                pattern_to_paths[pattern].append(current_path)
                return

            visited_ids.add(obj_id)
            try:
                # Check patterns against current node if its path was selected
                obj_type_name = getattr(type(current_obj), "__name__", "<unknown_type>")
                if current_path in path_to_plan:
                    for pattern, plan in self._patterns.items():
                        if plan is path_to_plan[current_path]:  # Check plan instance
                            if (
                                fnmatch.fnmatch(current_path, pattern)
                                or fnmatch.fnmatch(obj_type_name, pattern)
                            ) and current_path not in pattern_to_paths[pattern]:
                                pattern_to_paths[pattern].append(current_path)

                # Recurse using registered iterators
                iterator_func = self._get_iterator(current_obj)
                if iterator_func:
                    for key_or_index_str, child_item in iterator_func(current_obj):
                        if not isinstance(
                            key_or_index_str, str
                        ) or not key_or_index_str:
                            continue
                        child_path = f"{current_path}.{key_or_index_str}" if current_path else key_or_index_str
                        _find_matches(child_item, child_path)
            finally:
                visited_ids.remove(obj_id)

        _find_matches(obj, "")

        # Format the results
        lines = ["Pattern Matches:"]
        found_match = False
        for pattern in sorted(self._patterns.keys()):  # Consistent order
            paths = pattern_to_paths[pattern]
            if paths:
                found_match = True
                lines.append(f"Pattern: {pattern!r}")  # Use repr for pattern
                # Sort paths for consistent order
                for path in sorted(paths):
                    display_path = path if path else "<root>"
                    lines.append(f"  - {display_path}")
                lines.append("")  # Blank line between patterns

        if not found_match:
            lines.append("  (No matches found)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Standard representation showing class name and patterns."""
        # Sort patterns for consistent repr
        items_str = "\n"
        for pattern in sorted(self._patterns.keys()):
            plan = self._patterns[pattern]
            items_str += f"  {pattern!r}: {plan!r},\\n"  # Use !r for pattern and plan
        content = f"{{{items_str}}}" if items_str != "\n" else "{}"
        return f"{self.__class__.__name__}({content})"

    def __call__(self, obj: Any, *args: Any, **kwargs: Any) -> Any:
        """Shortcut to materialize and execute the selector."""
        return self.materialize(obj).execute(obj, *args, **kwargs)


# ==============================================================================
# Default Iterator Definitions and Registration
# ==============================================================================


@Selector.register_iterator(nn.Module)
def _nn_module_iterator(module: nn.Module) -> Iterator[Tuple[str, Any]]:
    """Yields ('name', child_module) for named children."""
    for name, child in module.named_children():
        yield name, child


# Register using the predicate dataclasses.is_dataclass
@Selector.register_iterator(dataclasses.is_dataclass)
def _dataclass_iterator(obj: Any) -> Iterator[Tuple[str, Any]]:
    """Yields ('field_name', field_value) for dataclass fields."""
    # Redundant check, but safe. Predicate already matched.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            try:
                # if field.repr: # Example check for field metadata
                value = getattr(obj, field.name)
                yield field.name, value
            except AttributeError:
                logging.debug(f"AttributeError accessing field {field.name} for {type(obj)}")


@Selector.register_iterator(list)
@Selector.register_iterator(tuple)
def _list_tuple_iterator(seq: Union[List, Tuple]) -> Iterator[Tuple[str, Any]]:
    """Yields (str(index), item) for list/tuple elements."""
    for i, item in enumerate(seq):
        yield str(i), item


@Selector.register_iterator(dict)
def _dict_iterator(d: Dict) -> Iterator[Tuple[str, Any]]:
    """Yields (str(key), value) for dict items."""
    for k, v in d.items():
        try:
            # Convert key to string for path segment
            key_str = str(k)
            # Basic check: Ensure str(k) doesn't create ambiguous dots.
            # More complex keys might still cause issues in parsing/applying.
            if "." in key_str or "[" in key_str or "]" in key_str:
                logging.warning(
                    f"Dict key {k!r} string form '{key_str}' contains '.', '[', or ']'. Path resolution might be ambiguous."
                )
            # Yield segment starting with a dot
            yield key_str, v
        except Exception as e:
            logging.debug(f"Cannot create path segment for dict key {k!r}: {e}")
