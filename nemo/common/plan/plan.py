from typing import (
    Self,
    TypeVar,
    Generic,
    Any,
    Iterator,
    Optional,
    Callable,
    Dict,
    Type,
    Literal,
    ClassVar,
    List,
)
from collections import OrderedDict
import threading
import functools
import logging
import os
import bdb
import traceback
import dataclasses
import inspect

import torch

from nemo.lightning.io.mixin import (
    _io_init,
    IOProtocol,
)

try:
    import fiddle._src.experimental.dataclasses as fdl_dc
except ImportError:
    # Fallback for when fiddle is not available or has a different structure
    fdl_dc = None

OutputT = TypeVar("OutputT")
ThreadedMode = Literal["sync", "async"]
PrimaryWorkerMode = Literal["skip", "sync", "broadcast"]


class PlanMeta(type):
    """Metaclass to configure Plan behaviors via class kwargs with a dynamic registry."""

    _behavior_registry: Dict[str, tuple[Callable, list[str]]] = {}

    @classmethod
    def register_behavior(cls, applicator: Callable) -> Callable:
        """Decorator to register a behavior applicator with its allowed values."""
        name = applicator.__name__
        allowed_values = getattr(applicator, "_allowed_values", [])
        if name in cls._behavior_registry:
            raise ValueError(f"Behavior '{name}' is already registered")
        cls._behavior_registry[name] = (applicator, allowed_values)
        return applicator

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs) -> Type:
        """Create a new Plan subclass with behavior validation and application."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip base Plan class
        if name == "Plan":
            return cls

        # Validate and store behaviors
        for behavior, (applicator, allowed_values) in mcs._behavior_registry.items():
            value = kwargs.get(behavior)
            if value and value not in allowed_values:
                raise ValueError(
                    f"Invalid {behavior}: {value}. Must be one of {allowed_values}"
                )
            setattr(cls, f"_{behavior}", value)

        # Decorate execute if defined
        if "execute" in namespace:
            execute = namespace["execute"]
            # Apply behaviors in registry order (could be made configurable)
            for behavior, (applicator, _) in mcs._behavior_registry.items():
                if behavior in kwargs:
                    execute = applicator(kwargs[behavior])(execute)
            setattr(cls, "execute", execute)

        # Apply custom Plan IO wrapping instead of standard NeMo wrapping
        # cls = _plan_io_wrap_init(cls)

        return cls


# Add this function to handle Plan-specific IO wrapping
def _plan_io_wrap_init(cls):
    """Wraps the __init__ method of a Plan class to add IO functionality that handles inheritance.

    The key insight is to only set the __io__ attribute if it doesn't already exist.
    This way, only the outermost __init__ in an inheritance chain will create the config,
    which avoids issues with arguments meant for parent classes.
    """
    original_init = cls.__init__

    if getattr(cls, "__wrapped_init__", False):
        return cls

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # First call the original init to ensure attributes are properly set
        original_init(self, *args, **kwargs)

        # Only create the IO config if it's not already set (by a parent class)
        if not hasattr(self, "__io__"):
            try:
                # Get the signature of this specific class's init
                sig = inspect.signature(cls.__init__)

                # Try binding with just the parameters this class's __init__ accepts
                parameters = list(sig.parameters.keys())

                # Filter kwargs to only include those in the signature
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in parameters}

                # Add positional args by position if they fit in the signature
                # Skip 'self' which is the first parameter
                other_params = (
                    parameters[1:]
                    if parameters and parameters[0] == "self"
                    else parameters
                )
                for i, arg in enumerate(args):
                    if i < len(other_params):
                        filtered_kwargs[other_params[i]] = arg

                # Now create the IO config with just the filtered parameters
                import pdb

                pdb.set_trace()

                self.__io__ = _io_init(self, **filtered_kwargs)
            except Exception as e:
                # If anything goes wrong, log it but don't break execution
                logging.debug(f"Failed to create IO config for {cls.__name__}: {e}")
                # Set a dummy IO attribute to prevent further attempts
                self.__io__ = None

    cls.__init__ = wrapped_init
    cls.__wrapped_init__ = True
    return cls


class PlanExecutionError(Exception):
    """Exception raised when a Plan execution fails, with context about the execution path.

    This exception tracks the full plan execution hierarchy, allowing for better debugging
    of complex DAG operations.

    Attributes:
        plan_path: List of (plan_name, plan_class, status, depth) tuples representing the execution path
        original_error: The original exception that was raised
        operation: The specific operation or method that failed (if available)
    """

    def __init__(
        self,
        plan: "Plan",
        original_error: Exception,
        plan_path: Optional[List[tuple[str, str, str, int]]] = None,
        operation: Optional[str] = None,
    ):
        self.plan_path = plan_path or []
        self.plan = plan  # Store the plan object for repr

        # Get plan name and class
        plan_name = plan.name or plan.__class__.__name__
        plan_class = plan.__class__.__name__

        # Check if the plan is already in the path
        plan_in_path = False
        for name, cls_name, _, _ in self.plan_path:
            if name == plan_name and cls_name == plan_class:
                plan_in_path = True
                break

        if not plan_in_path:
            # Determine depth - either 0 for root or increment from previous
            depth = 0
            if self.plan_path and len(self.plan_path[0]) >= 4:
                depth = self.plan_path[0][3] + 1

            # Add to the plan path
            self.plan_path.insert(0, (plan_name, plan_class, "❌", depth))

        self.original_error = original_error
        self.operation = operation

        # Try to extract operation from traceback if not provided
        if self.operation is None:
            self.operation = self._extract_operation_from_traceback()

        # Create the message with tree visualization
        message = self._format_error_tree()
        super().__init__(message)

    def _extract_operation_from_traceback(self) -> Optional[str]:
        """Extract the operation name from the traceback of the original error.

        This method walks the traceback to find the most relevant method call that caused the error.
        It prioritizes methods associated with the object that's mentioned in the error message.
        """
        try:
            tb = self.original_error.__traceback__
            if not tb:
                return None

            # Get the error message to help identify the relevant class
            err_msg = str(self.original_error)
            err_type = type(self.original_error).__name__

            # Extract class name from error message if possible
            # This helps identify methods on the object that actually caused the error
            mentioned_class = None
            missing_attr = None

            if err_type == "AttributeError" and "object has no attribute" in err_msg:
                # Attribute error pattern: "'ClassName' object has no attribute 'xyz'"
                parts = err_msg.split("'")
                if len(parts) >= 4:
                    mentioned_class = parts[1]
                    missing_attr = parts[3]

            # First pass: For attribute errors, try to find the specific method call
            if mentioned_class and missing_attr:
                current_tb = tb
                while current_tb:
                    frame = current_tb.tb_frame
                    func_name = frame.f_code.co_name

                    # Check if this is the execution frame for our mentioned class
                    if "self" in frame.f_locals:
                        instance = frame.f_locals["self"]
                        if instance.__class__.__name__ == mentioned_class:
                            # If it's trying to execute the missing method directly
                            if func_name == "execute":
                                # Look at the bytecode to find what method it's trying to call
                                import dis

                                try:
                                    # Get the bytecode for this frame
                                    code = frame.f_code
                                    # Get the line number where the error occurred
                                    error_lineno = current_tb.tb_lineno
                                    # Get the bytecode for this line
                                    for op in dis.get_instructions(code):
                                        if op.starts_line == error_lineno:
                                            # If it's a LOAD_ATTR or LOAD_METHOD, we found the call
                                            if op.opname in (
                                                "LOAD_ATTR",
                                                "LOAD_METHOD",
                                            ):
                                                # The argval is the attribute name being loaded
                                                return f"{mentioned_class}.execute() → {mentioned_class}.{op.argval}()"
                                except Exception:
                                    pass  # If we can't analyze bytecode, fall back to simpler info

                            # If it's trying to access the missing attribute
                            if func_name == "__getattr__" and "name" in frame.f_locals:
                                if frame.f_locals["name"] == missing_attr:
                                    # We found the frame where the attribute was requested
                                    # Now backtrack to find who called this method
                                    prev_tb = current_tb.tb_next
                                    if prev_tb:
                                        caller_frame = prev_tb.tb_frame
                                        caller_func = caller_frame.f_code.co_name
                                        if "self" in caller_frame.f_locals:
                                            caller_class = caller_frame.f_locals[
                                                "self"
                                            ].__class__.__name__
                                            return f"{caller_class}.{caller_func}() → {mentioned_class}.{missing_attr}()"

                    current_tb = current_tb.tb_next

            # Second pass: find the most specific/relevant method
            # Skip common wrapper methods and prioritize methods from implementation files
            operation_candidates = []
            current_tb = tb
            while current_tb:
                frame = current_tb.tb_frame
                if "self" in frame.f_locals:
                    method_name = frame.f_code.co_name
                    if method_name not in (
                        "execute",
                        "__call__",
                        "_call_impl",
                        "wrapper",
                    ):
                        instance = frame.f_locals["self"]
                        class_name = instance.__class__.__name__

                        # Check if the method is defined in our implementation files (not in plan.py)
                        filename = frame.f_code.co_filename
                        if "plan.py" not in filename:
                            # Higher priority for methods in implementation files
                            return f"{class_name}.{method_name}()"

                        # Add as a candidate
                        operation_candidates.append((class_name, method_name))
                current_tb = current_tb.tb_next

            # Return the first candidate if any were found
            if operation_candidates:
                class_name, method_name = operation_candidates[0]
                return f"{class_name}.{method_name}()"

            return None
        except Exception:
            # If anything goes wrong, we don't want to crash during error reporting
            return None

    def _format_error_tree(self) -> str:
        """Format the error as a tree visualization with error highlighting."""
        if not self.plan_path:
            return f"ERROR: {self.original_error}"

        # Find the outermost plan (without parent)
        root_plan = self.plan
        while (
            hasattr(root_plan, "_parent")
            and getattr(root_plan, "_parent", None) is not None
        ):
            root_plan = root_plan._parent

        # Group items by depth for proper tree structure
        by_depth = {}
        for item in self.plan_path:
            if len(item) >= 4:  # Make sure we have depth info
                name, cls_name, status, depth = item
                if depth not in by_depth:
                    by_depth[depth] = []
                by_depth[depth].append((name, cls_name, status))

        # Start with the root plan (outermost plan)
        lines = []

        # Use the root plan's repr
        if root_plan is not None:
            lines.append(repr(root_plan))
        elif 0 in by_depth and by_depth[0]:
            root_name, root_class, _ = by_depth[0][0]
            lines.append(f"{root_class}({root_name})")
        else:
            # Fallback if we don't have proper depth info
            root_name, root_class, _ = self.plan_path[0][:3]  # First 3 elements
            lines.append(f"{root_class}({root_name})")

        # Build the tree
        max_depth = max(by_depth.keys()) if by_depth else 0

        # Track which depths still need vertical connectors
        active_branches = set()

        # Process each depth level after the root
        for depth in range(1, max_depth + 1):
            if depth not in by_depth:
                continue

            items = by_depth[depth]

            for i, (name, cls_name, status) in enumerate(items):
                is_last = i == len(items) - 1

                # Build the prefix based on depth and active branches
                prefix = ""
                for level in range(depth):
                    if level < depth - 1:
                        # Add vertical connector or space for previous levels
                        if level in active_branches:
                            prefix += "│   "
                        else:
                            prefix += "    "
                    else:
                        # For current level
                        if is_last:
                            prefix += "└── "  # Last item at this level
                            # Remove this branch from active ones
                            if level in active_branches:
                                active_branches.remove(level)
                        else:
                            prefix += "├── "  # More items follow
                            # Add to active branches
                            active_branches.add(level)

                # Add the plan entry
                lines.append(f"{prefix}{cls_name}: {name} {status}")

        # For the deepest level (error information), use consistent indentation
        last_item_prefix = ""
        for level in range(max_depth):
            if level in active_branches:
                last_item_prefix += "│   "
            else:
                last_item_prefix += "    "

        # Now add the error details with proper indentation
        if self.operation:
            lines.append(f"{last_item_prefix}    └── Operation: {self.operation}")

        # Add the error message
        error_type = self.original_error.__class__.__name__
        error_msg = str(self.original_error)

        # Preserve special messages like "Did you mean"
        if "Did you mean" in error_msg:
            parts = error_msg.split("Did you mean")
            lines.append(
                f"{last_item_prefix}    └── ERROR: {error_type}: {parts[0].strip()}"
            )
            lines.append(f"{last_item_prefix}        Did you mean{parts[1]}")
        else:
            lines.append(f"{last_item_prefix}    └── ERROR: {error_type}: {error_msg}")

        # Add a fix tip if available
        fix_tip = self.get_fix_tip()
        if fix_tip:
            lines.append("")
            lines.append(f"💡 FIX TIP: {fix_tip}")

        return "\n".join(lines)

    def detailed_traceback(self) -> str:
        """Generate a detailed traceback that includes the plan execution path.

        This method combines the plan hierarchy tree with a clean version of the original
        exception's traceback, avoiding the nested exception chains.

        Returns:
            A formatted string with both plan hierarchy and a clean exception traceback
        """
        # Start with the header for the plan hierarchy
        lines = ["============================================================"]
        lines.append("Plan Traceback:")
        lines.append("============================================================")
        lines.append(self._format_error_tree())
        lines.append("")

        # Add a separator
        lines.append("=" * 60)
        lines.append("Full Python Traceback:")
        lines.append("=" * 60)

        # Add the full original Python traceback
        lines.append(self.original_traceback())

        return "\n".join(lines)

    def _extract_relevant_frames(self) -> List[str]:
        """Extract the most relevant frames from the traceback.

        This method simplifies the traceback by:
        1. Retaining key frames that show the error propagation path
        2. Ensuring framework code doesn't overwhelm user code
        3. Focusing on the frames that directly led to the error

        Returns:
            List of formatted frame information strings
        """
        frames = []

        # Get the traceback frames
        tb = self.original_error.__traceback__
        if not tb:
            return [f"{type(self.original_error).__name__}: {str(self.original_error)}"]

        # Collect all frames
        frame_info = []
        internal_patterns = [
            "site-packages",
            "lib/python",
        ]  # Patterns for external library code

        while tb:
            frame = tb.tb_frame
            filename = frame.f_code.co_filename
            lineno = tb.tb_lineno
            co_name = frame.f_code.co_name

            # Get shortened filename (just the last 2-3 parts)
            parts = filename.split("/")
            short_filename = "/".join(parts[-3:]) if len(parts) > 3 else filename

            # Keep track if this is user code or library code
            is_external = any(pattern in filename for pattern in internal_patterns)

            # Build the frame info - include a marker for external code
            frame_info.append((short_filename, lineno, co_name, is_external))
            tb = tb.tb_next

        # Identify important frames
        important_frames = []
        user_code_count = 0

        # Count user code frames
        for _, _, _, is_external in frame_info:
            if not is_external:
                user_code_count += 1

        # If we have few user code frames, include all frames
        # Otherwise, prioritize user code but include some library frames for context
        for filename, lineno, co_name, is_external in reversed(frame_info):
            # Always include user code frames
            if not is_external:
                important_frames.append((filename, lineno, co_name))
            # Include external frames if we don't have many user frames
            elif user_code_count < 5 or len(important_frames) < 10:
                important_frames.append((filename, lineno, co_name))

        # Format the important frames
        for filename, lineno, co_name in important_frames:
            frames.append(f'  File "{filename}", line {lineno}, in {co_name}')

        # Add the error message at the end
        error_type = type(self.original_error).__name__
        error_msg = str(self.original_error)
        frames.append(f"{error_type}: {error_msg}")

        return frames

    def print_detailed_traceback(self) -> None:
        """Print the detailed traceback to stderr."""
        import sys

        print(self.detailed_traceback(), file=sys.stderr)

    def original_traceback(self) -> str:
        """Get the original Python traceback with exception chaining.

        This method returns the standard Python traceback with all exception chaining intact.

        Returns:
            The standard Python traceback as a string
        """
        # Add the original traceback
        if self.original_error.__traceback__:
            return "".join(
                traceback.format_exception(
                    type(self.original_error),
                    self.original_error,
                    self.original_error.__traceback__,
                )
            )
        else:
            return f"{type(self.original_error).__name__}: {str(self.original_error)}"

    def get_fix_tip(self) -> Optional[str]:
        """Extract a helpful suggestion for fixing the error if possible.

        Analyzes the error message to find hints about how to fix it, like
        "Did you mean: 'xyz'" suggestions or other common patterns.

        Returns:
            A string with a fix suggestion, or None if no tip could be extracted
        """
        error_msg = str(self.original_error)

        # Look for "Did you mean" suggestions
        if "Did you mean" in error_msg:
            parts = error_msg.split("Did you mean")
            if len(parts) >= 2:
                # Extract the suggestion
                suggestion = parts[1].strip()
                if suggestion.startswith(":"):
                    suggestion = suggestion[1:].strip()
                if suggestion.startswith("'") and "'" in suggestion[1:]:
                    # Extract the suggested name
                    suggested_name = suggestion.split("'")[1]
                    # Get the operation if available
                    op_info = ""
                    if self.operation:
                        op_parts = self.operation.split(".")
                        if len(op_parts) >= 2:
                            class_name = op_parts[0]
                            wrong_method = op_parts[1].rstrip("()")
                            op_info = f" in {class_name}"

                    return f"Try using '{suggested_name}' instead of '{wrong_method}'{op_info}"
                return f"Suggestion: Did you mean{suggestion}"

        # Look for missing file errors
        if isinstance(self.original_error, FileNotFoundError):
            path = error_msg.replace("Checkpoint file not found: ", "").strip()
            return f"The file '{path}' doesn't exist. Check if the path is correct or if you need to create this file first."

        # Add more patterns as needed

        return None


class Plan(Generic[OutputT], metaclass=PlanMeta):
    """Base class for creating executable plans with hierarchical composition.

    A Plan represents a unit of execution that can contain other Plans as children.
    Plans execute in sequence, with each child's output optionally feeding into the next.
    Plans can be composed hierarchically and support pattern-based selection of components.

    Class-level behaviors can be specified using keyword arguments:
    - threaded: "sync" or "async" - Controls whether execution runs in a separate thread
    - primary_worker_only: "skip", "sync", or "broadcast" - Controls distributed execution
    - child_policy: "inherit" or "isolate" - Controls how behaviors propagate to child plans
      With "inherit" (default), child plans inherit parent's execution context
      With "isolate", child plans always apply their own behaviors regardless of parent

    Example:
        ```python
        # Define a Plan with class-level behaviors
        class MyPlan(Plan, threaded="async", primary_worker_only="skip", child_policy="inherit"):
            def execute(self, *args, **kwargs):
                # Custom execution logic
                pass

        # Create a simple plan with two children
        plan = Plan(
            first_step=SomeChildPlan(),
            second_step=AnotherChildPlan()
        )

        # Execute the plan
        result = plan(input_data)
        ```

    Attributes:
        name (str | None): Optional name identifier for the plan.
        _children_dict (OrderedDict[str, Plan]): Ordered dictionary of child plans.
    """

    # Class variables to help with auto-completion
    threaded: ClassVar[ThreadedMode] = None  # type: ignore
    primary_worker_only: ClassVar[PrimaryWorkerMode] = None  # type: ignore

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize a plan with optional pattern-based sub-plans.

        Args:
            *args: Positional arguments that may be Plan instances or dictionaries to be
                wrapped in Selector plans. Dictionary arguments are automatically converted
                to Selector plans.
            **kwargs: Keyword arguments where values may be Plan instances or dictionaries
                to be wrapped in Selector plans. Dictionary values are automatically
                converted to Selector plans.

        Raises:
            TypeError: If any argument is not a Plan instance or dictionary.
        """
        from nemo.common.plan.selector import Selector
        
        self._name: str | None = None
        self._children_dict: OrderedDict[str, "Plan"] = OrderedDict()

        # Add args as children in order
        for i, arg in enumerate(args):
            if isinstance(arg, dict):
                # Create a selector for the patterns
                selector = Selector(arg)
                self._children_dict[str(i)] = selector
            elif isinstance(arg, Plan):
                self._children_dict[str(i)] = arg
            else:
                raise TypeError(f"Expected Plan or dict, got {type(arg)}")

        # Add kwargs as children
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Create a selector for the patterns
                selector = Selector(value)
                self._children_dict[key] = selector
            elif isinstance(value, Plan):
                self._children_dict[key] = value
            else:
                raise TypeError(f"Expected Plan or dict, got {type(value)}")

    @property
    def _children(self) -> OrderedDict[str, "Plan"]:
        """Get the children dictionary, initializing it if needed."""
        if not hasattr(self, "_children_dict"):
            self._children_dict = OrderedDict()
        return self._children_dict

    @property
    def name(self) -> str | None:
        if not hasattr(self, "_name"):
            self._name = None
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Plan):
            # Use object.__setattr__ to avoid recursion when setting parent
            object.__setattr__(value, "_parent", self)
            object.__setattr__(value, "_name", name)
            self._children[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._children:
            child = self._children[name]
            # Use object.__delattr__ to avoid recursion when removing parent
            if hasattr(child, "_parent"):
                object.__delattr__(child, "_parent")
            del self._children[name]
        else:
            super().__delattr__(name)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute, checking children first if it's not a special attribute."""
        if name.startswith("_"):
            return super().__getattr__(name)
        if name in self._children:
            return self._children[name]

        ## Safely try to find the attribute in parent classes
        try:
            # Use object.__getattribute__ to bypass our __getattr__
            return super().__getattribute__(name)
        except AttributeError:
            # Let the execute method handle wrapping the error
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __call__(self, *args: Any, **kwargs: Any) -> OutputT:
        """Call the plan, which executes it and its children in order.

        Args:
            *args: Arguments to pass to the first child plan
            **kwargs: Keyword arguments to pass to each child plan

        Returns:
            The result of the last child's execution, or the first argument if no children
        """
        try:
            return self.execute(*args, **kwargs)
        except Exception as e:
            # Let the exception propagate up through the plan hierarchy
            # Only format and exit at the outermost plan
            if not hasattr(self, "_parent"):
                # We're at the root plan, format and exit
                import sys

                if isinstance(e, PlanExecutionError):
                    print(e.detailed_traceback(), file=sys.stderr)
                else:
                    formatted_error = Plan.format_exception(e)
                    print(formatted_error, file=sys.stderr)
                sys.exit(1)
            raise  # Re-raise to propagate up to parent plan

    def _get_name(self) -> str:
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """Set the extra representation of the module.

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for name, plan in self._children.items():
            mod_str = repr(plan)
            mod_str = self._addindent(mod_str, 2)
            child_lines.append(f"({name}): {mod_str}")
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def _addindent(self, s: str, numSpaces: int) -> str:
        s = s.split("\n")
        if len(s) == 1:
            return s[0]
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    def children(self) -> Iterator["Plan"]:
        """Return an iterator over immediate children plans.

        Returns:
            An iterator over immediate children plans.
        """
        return iter(self._children.values())

    def named_children(self) -> Iterator[tuple[str, "Plan"]]:
        """Return an iterator over immediate children plans, yielding both the name of the plan as well as the plan itself.

        Returns:
            An iterator over immediate children plans, yielding both the name of the plan as well as the plan itself.
        """
        return iter(self._children.items())

    def execute(self, *args: Any, **kwargs: Any) -> OutputT:
        """Execute the plan and all its children in sequence.

        This method implements the core execution logic for a Plan:
        1. If there are no children, returns the first argument or None
        2. For each child plan in sequence:
           - Executes the child with current arguments
           - If child returns non-None, uses that as first argument for next child
           - Passes through remaining args and all kwargs unchanged
        3. Returns the final result (output of last child or first arg if no children)

        Args:
            *args: Positional arguments to pass to the first child plan. The first
                argument is treated specially as it may be modified/replaced by child
                plan outputs.
            **kwargs: Keyword arguments passed unchanged to all child plans.

        Returns:
            OutputT: The result of the last child's execution, or the first argument if
                no children, or None if no children and no arguments.

        Raises:
            PlanExecutionError: If execution of any child plan fails, with context
                about the execution path.

        Example:
            ```python
            plan = Plan(step1=ProcessingPlan(), step2=TransformPlan())

            # Execute with initial input
            result = plan.execute(initial_data, extra_arg=True)
            ```
        """
        if not self._children:
            try:
                # Execute the plan's implementation if it exists
                return (
                    self._execute_impl(*args, **kwargs)
                    if hasattr(self, "_execute_impl")
                    else args[0]
                    if args
                    else None
                )
            except Exception as e:
                # If no children, wrap the exception with this plan's path
                path = [
                    (
                        self.name or self.__class__.__name__,
                        self.__class__.__name__,
                        "❌",
                        0,
                    )
                ]
                raise PlanExecutionError(self, e, path)

        # Start with the initial arguments
        current_args = args
        current_kwargs = kwargs

        # Track successful children for error reporting
        successful_children = []

        # Execute each child in order, passing its output to the next
        for child_name, child in self._children.items():
            try:
                result = child(*current_args, **current_kwargs)
                # If result is None, keep the previous input, otherwise use the result
                if result is not None:
                    current_args = (
                        (result,) + current_args[1:] if current_args else (result,)
                    )
                current_kwargs = kwargs

                # Record this child as successful with depth 1 (direct child)
                successful_children.append(
                    (child_name, child.__class__.__name__, "✓", 1)
                )

            except PlanExecutionError as e:
                # Create a new path starting with successful children
                path = list(successful_children)

                # Add the current failed child with depth 1 (direct child)
                path.append((child_name, child.__class__.__name__, "❌", 1))

                # Adjust the depth of the nested path items to reflect their level in the hierarchy
                nested_path = []
                for item in e.plan_path:
                    if len(item) >= 4:
                        # Increment the depth to reflect nesting
                        name, cls_name, status, depth = item
                        nested_path.append((name, cls_name, status, depth + 1))
                    else:
                        # Handle older format (backward compatibility)
                        nested_path.append(item + (2,))  # Add depth 2 as default

                # Add the nested path to our path
                path.extend(nested_path)

                # Re-raise with the complete path
                raise PlanExecutionError(self, e.original_error, path, e.operation)

            except Exception as e:
                # New exception, create a fresh PlanExecutionError with the complete path
                if not isinstance(e, bdb.BdbQuit):  # Don't wrap debugger exit
                    # Build path with successful children and this failed child
                    path = list(successful_children)
                    path.append((child_name, child.__class__.__name__, "❌", 1))
                    raise PlanExecutionError(self, e, path)
                raise  # Re-raise debugger exceptions without wrapping

        return current_args[0]

    def revert(self) -> None:
        """Revert the plan and all its children in reverse order.

        This method provides rollback functionality by reverting all changes made by
        child plans in reverse order of execution. Each child plan must implement
        its own revert() method for this to work.

        Raises:
            NotImplementedError: If any child plan does not implement revert().

        Example:
            ```python
            try:
                plan.execute(data)
            except Exception:
                plan.revert()  # Roll back any changes
                raise
            ```
        """
        # Check if all children are revertible
        non_revertible = []
        for name, child in self._children.items():
            if not hasattr(child, "revert"):
                non_revertible.append(name)

        if non_revertible:
            raise NotImplementedError(
                f"Children {non_revertible} are not revertible. "
                "Subclasses must implement revert() to support rollback."
            )

        # Revert children in reverse order
        for child in reversed(list(self._children.values())):
            child.revert()

    def apply(self, fn: Any) -> Self:
        """Apply fn recursively to every subplan (as returned by .children()) as well as self.

        Args:
            fn: Function to be applied to each subplan.

        Returns:
            The plan itself.
        """
        for child in self._children.values():
            child.apply(fn)
        fn(self)
        return self

    def apply_to_children(self, fn: Any) -> Self:
        """Apply fn recursively to every subplan (as returned by .children()) but not self.

        Args:
            fn: Function to be applied to each subplan.

        Returns:
            The plan itself.
        """
        for child in self._children.values():
            child.apply(fn)
        return self

    def wait(self):
        """Wait for the asynchronous execution to complete and return the result.

        Raises:
            RuntimeError: If no asynchronous execution is in progress.
        """
        if not hasattr(self, "_future") or not self._future:
            raise RuntimeError("No async execution in progress")
        return self._future.result()

    def __init_subclass__(
        cls,
        *,
        threaded: Optional[ThreadedMode] = None,
        primary_worker_only: Optional[PrimaryWorkerMode] = None,
        **kwargs: Any,
    ) -> None:
        """Enable autocomplete for known behaviors."""
        super().__init_subclass__()
        # Optionally store for introspection
        cls.threaded = threaded
        cls.primary_worker_only = primary_worker_only

    def visualize(self, indent: int = 0) -> str:
        """Generate a tree visualization of the plan hierarchy.

        This method produces a string representation of the entire plan execution
        hierarchy, making it easier to understand complex plan structures before execution.

        Args:
            indent: Starting indentation level

        Returns:
            A formatted string showing the plan hierarchy
        """
        lines = []

        # Get the plan name and class
        plan_name = self.name or ""
        plan_class = self.__class__.__name__
        if plan_name:
            header = f"{plan_class}({plan_name})"
        else:
            header = f"{plan_class}()"

        # Add any extra representation
        extra = self.extra_repr()
        if extra:
            header = f"{header} - {extra}"

        lines.append(" " * indent + header)

        # Add child plans
        for name, child in self._children.items():
            # Add this child with increased indentation
            child_indent = indent + 2
            if isinstance(child, Plan):
                # Recursive visualization for Plan objects
                child_viz = child.visualize(child_indent)
                # Replace the first line with our custom header
                child_lines = child_viz.split("\n")
                child_class = child.__class__.__name__
                if len(child_lines) > 1:
                    # Has multiple lines, keep the tree structure
                    child_header = " " * child_indent + f"├── ({name}): {child_class}"
                    lines.append(child_header)
                    lines.extend(
                        [
                            " " * child_indent + "│   " + line.strip()
                            for line in child_lines
                        ]
                    )
                else:
                    # Single line, simpler output
                    child_header = " " * child_indent + f"└── ({name}): {child_class}"
                    lines.append(child_header)
            else:
                # Simple entry for non-Plan objects
                lines.append(" " * child_indent + f"└── ({name}): {repr(child)}")

        return "\n".join(lines)

    def print_plan(self) -> None:
        """Print the plan hierarchy as a tree visualization.

        This is a convenience method that prints the output of visualize().
        """
        print(self.visualize())

    @staticmethod
    def format_exception(exc: Exception) -> str:
        """Format an exception with enhanced context if it's a PlanExecutionError.

        Args:
            exc: The exception to format

        Returns:
            A formatted string representation of the exception
        """
        if isinstance(exc, PlanExecutionError):
            return exc.detailed_traceback()
        else:
            # Try to extract better context about the plan hierarchy from the traceback
            operation = None
            error_plan = None
            plan_path = []

            # Walk the traceback to find plans
            tb = exc.__traceback__

            # First pass: collect all plans in the traceback
            plan_frames = []
            while tb:
                frame = tb.tb_frame
                if "self" in frame.f_locals:
                    instance = frame.f_locals["self"]
                    # Check if this is a Plan instance
                    if isinstance(instance, Plan):
                        # Record the plan with frame info
                        method_name = frame.f_code.co_name
                        filename = frame.f_code.co_filename
                        lineno = tb.tb_lineno
                        plan_frames.append((instance, method_name, filename, lineno))
                tb = tb.tb_next

            # The error happened in the first plan we encounter in the traceback
            if plan_frames:
                error_plan, error_method, _, _ = plan_frames[0]
                if error_method not in ("execute", "__call__", "_call_impl", "wrapper"):
                    operation = f"{error_plan.__class__.__name__}.{error_method}()"

            # If we couldn't find any plans in the traceback, fall back to a dummy plan
            if not error_plan:
                dummy_plan = Plan()
                dummy_plan.name = "ErrorSource"
                plan_error = PlanExecutionError(dummy_plan, exc)
                return plan_error.detailed_traceback()

            # Create a map of all plans in the hierarchy
            all_plans = {}
            to_explore = [error_plan]  # Start with the error plan

            # Helper function to explore a plan's hierarchy
            def explore_plan(plan: Plan, parent_id=None, depth=0):
                plan_id = id(plan)

                # If we've already seen this plan, just update its parent if needed
                if plan_id in all_plans:
                    plan_info = all_plans[plan_id]
                    if parent_id is not None and parent_id not in plan_info["parents"]:
                        plan_info["parents"].append(parent_id)
                    return

                # Store basic plan info
                all_plans[plan_id] = {
                    "plan": plan,
                    "class": plan.__class__.__name__,
                    "name": plan.name or "",
                    "parents": [parent_id] if parent_id else [],
                    "depth": depth,
                    "error": plan is error_plan,
                }

                # Explore the plan's children
                if hasattr(plan, "_children_dict"):
                    for child_name, child_plan in plan._children_dict.items():
                        if isinstance(child_plan, Plan):
                            # Update the child's name with its key in the parent
                            if not child_plan.name:
                                child_plan.name = child_name
                            # Add to exploration queue
                            to_explore.append(child_plan)
                            # Recursively explore the child
                            explore_plan(child_plan, plan_id, depth + 1)

            # Start exploring from the error plan
            for plan in to_explore:
                explore_plan(plan)

            # Now find the root plan(s) (those with no parents)
            root_plans = [
                p_id for p_id, info in all_plans.items() if not info["parents"]
            ]

            # If we have multiple roots or can't find a clear root, scan outward from error plan
            if len(root_plans) != 1:
                # Try to find the "highest" parent by walking up from the error plan
                current_plan_id = id(error_plan)
                visited = set()

                while current_plan_id not in visited:
                    visited.add(current_plan_id)
                    # If this plan has no parents, it's a root
                    if not all_plans[current_plan_id]["parents"]:
                        root_plans = [current_plan_id]
                        break
                    # Move to the first parent
                    if all_plans[current_plan_id]["parents"]:
                        current_plan_id = all_plans[current_plan_id]["parents"][0]

            # If we still don't have a clear root, use the plan with lowest depth
            if not root_plans:
                min_depth = float("inf")
                root_id = None
                for p_id, info in all_plans.items():
                    if info["depth"] < min_depth:
                        min_depth = info["depth"]
                        root_id = p_id
                if root_id:
                    root_plans = [root_id]

            # Get the main root plan if we found one
            root_plan_id = root_plans[0] if root_plans else id(error_plan)

            # Build a tree showing the path from root to error plan
            plan_tree = []

            # Start with the root
            root_info = all_plans[root_plan_id]
            plan_tree.append(
                {
                    "plan_id": root_plan_id,
                    "name": root_info["name"] or root_info["class"].lower(),
                    "class": root_info["class"],
                    "depth": 0,
                    "error": root_info["error"],
                }
            )

            # Helper function to find children of a plan
            def get_children(parent_id):
                children = []
                for p_id, info in all_plans.items():
                    if parent_id in info["parents"]:
                        children.append(p_id)
                return children

            # Helper function to find a path from one plan to another
            def find_path(start_id, target_id, path=None, visited=None):
                if path is None:
                    path = []
                if visited is None:
                    visited = set()

                if start_id == target_id:
                    return path + [start_id]

                if start_id in visited:
                    return None

                visited.add(start_id)

                # Try children first
                children = get_children(start_id)
                for child_id in children:
                    new_path = find_path(
                        child_id, target_id, path + [start_id], visited
                    )
                    if new_path:
                        return new_path

                # Then try parents
                if all_plans[start_id]["parents"]:
                    for parent_id in all_plans[start_id]["parents"]:
                        new_path = find_path(
                            parent_id, target_id, path + [start_id], visited
                        )
                        if new_path:
                            return new_path

                return None

            # Find path from root to error plan
            error_path = find_path(root_plan_id, id(error_plan))

            # If we couldn't find a path, use direct parent-child relationships
            if not error_path:
                # Add immediate children of root
                children = get_children(root_plan_id)
                for child_id in children:
                    child_info = all_plans[child_id]
                    plan_tree.append(
                        {
                            "plan_id": child_id,
                            "name": child_info["name"] or child_info["class"].lower(),
                            "class": child_info["class"],
                            "depth": 1,
                            "error": child_info["error"],
                        }
                    )

                # Make sure error plan is included
                error_included = any(
                    node["plan_id"] == id(error_plan) for node in plan_tree
                )
                if not error_included:
                    error_info = all_plans[id(error_plan)]
                    plan_tree.append(
                        {
                            "plan_id": id(error_plan),
                            "name": error_info["name"] or error_info["class"].lower(),
                            "class": error_info["class"],
                            "depth": len(plan_tree),  # Add at the end
                            "error": True,
                        }
                    )
            else:
                # Use the path we found to build the tree
                for i, plan_id in enumerate(
                    error_path[1:], 1
                ):  # Skip root, already added
                    plan_info = all_plans[plan_id]
                    plan_tree.append(
                        {
                            "plan_id": plan_id,
                            "name": plan_info["name"] or plan_info["class"].lower(),
                            "class": plan_info["class"],
                            "depth": i,
                            "error": plan_info["error"],
                        }
                    )

            # Sort by depth to ensure proper tree structure
            plan_tree.sort(key=lambda x: x["depth"])

            # Convert to the format expected by PlanExecutionError
            for node in plan_tree:
                status = "❌" if node["error"] else "✓"
                # Don't mark later nodes as successful if an earlier one failed
                if any(n["error"] for n in plan_tree if n["depth"] < node["depth"]):
                    status = "❌"  # Mark as failed if any ancestor failed
                plan_path.append((node["name"], node["class"], status, node["depth"]))

            # Create a PlanExecutionError with the reconstructed path
            plan_error = PlanExecutionError(error_plan, exc, plan_path, operation)
            return plan_error.detailed_traceback()

    @staticmethod
    def run_with_clean_error(
        plan_factory: Callable[[], "Plan"], *args, **kwargs
    ) -> Any:
        """Execute a plan factory function with clean error handling.

        This method executes the provided plan factory and catches any exceptions,
        formatting them using our enhanced error reporting. It will print the clean
        error message and exit with a non-zero status code.

        Args:
            plan_factory: A function that returns a Plan to execute
            *args: Arguments to pass to the plan
            **kwargs: Keyword arguments to pass to the plan

        Returns:
            The result of the plan execution if successful

        Example:
            ```python
            # Instead of:
            # model = AutoModel("meta-llama/Llama-3.2-1B", convert="megatron", setup=trainer)

            # Use:
            model = Plan.run_with_clean_error(
                lambda: AutoModel("meta-llama/Llama-3.2-1B", convert="megatron", setup=trainer)
            )
            ```
        """
        try:
            # Create and execute the plan
            plan = plan_factory()
            return plan(*args, **kwargs)
        except Exception as e:
            # Format and print the error
            import sys

            formatted_error = Plan.format_exception(e)
            print(formatted_error, file=sys.stderr)

            # Print a tip about wrapping in a try-except if needed
            if not isinstance(e, PlanExecutionError):
                print(
                    "\nTip: To handle this error programmatically, wrap the call in a try-except block.",
                    file=sys.stderr,
                )

            # Exit with error code (this will terminate the program)
            sys.exit(1)

    @staticmethod
    def plan_io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
        """Plan-specific transform for I/O args that handles inheritance better.

        This method provides a custom implementation that skips arguments that
        are meant for parent classes but not present in the child class signature.
        """
        import inspect

        # Get the signature of the function
        sig = inspect.signature(init_fn)

        try:
            # Try binding the arguments to the function signature
            bound_args = sig.bind_partial(self, *args, **kwargs)
            config_kwargs = {
                k: v for k, v in bound_args.arguments.items() if k != "self"
            }
        except TypeError:
            # If binding fails, it's likely because args are meant for a parent class
            # In this case, only keep explicitly declared parameters in the signature
            param_names = [p for p in sig.parameters if p != "self"]
            config_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            # Add positional args based on signature order
            for i, arg in enumerate(args):
                if i < len(param_names):  # Only include args that match the signature
                    config_kwargs[param_names[i]] = arg

        # The rest of the processing is the same as in _io_transform_args
        to_del = []
        for key in config_kwargs:
            if isinstance(config_kwargs[key], IOProtocol):
                config_kwargs[key] = config_kwargs[key].__io__
            if hasattr(dataclasses, "is_dataclass") and dataclasses.is_dataclass(
                config_kwargs[key]
            ):
                config_kwargs[key] = fdl_dc.convert_dataclasses_to_configs(
                    config_kwargs[key], allow_post_init=True
                )
            if (
                hasattr(config_kwargs[key], "__class__")
                and config_kwargs[key].__class__.__name__
                == "_HAS_DEFAULT_FACTORY_CLASS"
            ):
                to_del.append(key)

        for key in to_del:
            del config_kwargs[key]

        return config_kwargs

    @property
    def root(self) -> "Plan":
        while hasattr(self, "_parent"):
            self = self._parent
        return self


def plan_execution_behavior(allowed_values: list[str]) -> Callable:
    """Decorator factory to mark a plan execution behavior with its allowed values."""

    def decorator(func: Callable) -> Callable:
        func._allowed_values = allowed_values
        return PlanMeta.register_behavior(func)

    return decorator


def _get_local_rank() -> int:
    """Get local rank based on available CUDA devices.

    Returns:
        int: Local rank determined from CUDA device mapping or environment variable
    """
    if not torch.cuda.is_available():
        return 0

    # First try to get from environment (most explicit)
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    # Then try to get from CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if devices:
            return devices.index(str(torch.cuda.current_device()))

    # Finally fall back to current device
    return torch.cuda.current_device()


@plan_execution_behavior(["skip", "sync", "broadcast"])
def primary_worker_only(mode: PrimaryWorkerMode = "sync") -> Callable:
    """Apply primary worker behavior to a method.

    For 'broadcast' mode, the Plan subclass must implement a 'broadcast' method that takes
    the result and returns the broadcasted value to all workers.

    Args:
        mode: "skip" (rank 0 only), "sync" (rank 0 with barriers), "broadcast" (rank 0 with custom broadcast).
              Defaults to "sync".
    """

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not torch.distributed.is_initialized():
                return method(self, *args, **kwargs)

            rank = torch.distributed.get_rank()
            local_rank = _get_local_rank()
            # torch.cuda.set_device(local_rank)

            if mode == "skip":
                return method(self, *args, **kwargs) if rank == 0 else None
            elif mode == "sync":
                torch.distributed.barrier(device_ids=[local_rank])
                result = method(self, *args, **kwargs) if rank == 0 else None
                torch.distributed.barrier(device_ids=[local_rank])
                return result
            elif mode == "broadcast":
                torch.distributed.barrier(device_ids=[local_rank])
                result = method(self, *args, **kwargs) if rank == 0 else None
                torch.distributed.barrier(device_ids=[local_rank])
                if not hasattr(self, "broadcast"):
                    raise NotImplementedError(
                        "Plan subclass must implement 'broadcast' method for 'broadcast' mode"
                    )
                return self.broadcast(result)

        return wrapper

    return decorator


@plan_execution_behavior(["sync", "async"])
def threaded(mode: ThreadedMode = "sync") -> Callable:
    """Apply threading behavior to a method."""
    is_async = mode == "async"

    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            # Check if we're in debug mode - either explicitly or via sys.gettrace()
            import sys

            debug_mode = (
                os.environ.get("AUTOMODEL_DEBUG", "0") == "1"
                or sys.gettrace() is not None
            )

            # If in debug mode, run directly without threading
            if debug_mode:
                return method(self, *args, **kwargs)

            if not hasattr(self, "_thread"):
                self._thread = None
                self._thread_result = None
                self._thread_exception = None

            def thread_target():
                original_env = os.environ.copy()
                try:
                    # Create isolated environment for the thread
                    thread_env = os.environ.copy()
                    # thread_env.update({
                    #     "WORLD_SIZE": "1",
                    #     "LOCAL_RANK": "0",
                    #     "RANK": "0",
                    #     "LOCAL_WORLD_SIZE": "1"
                    # })

                    # Set the new environment
                    os.environ.clear()
                    os.environ.update(thread_env)

                    self._thread_result = method(self, *args, **kwargs)
                except Exception as e:
                    # Don't store BdbQuit exceptions - these need special handling
                    if isinstance(e, bdb.BdbQuit):
                        raise
                    self._thread_exception = e
                finally:
                    # Restore original environment
                    os.environ.clear()
                    os.environ.update(original_env)

            if is_async and self._thread and self._thread.is_alive():
                # Cancel previous thread if still running
                self._thread = None

            # Use daemon=True to ensure thread doesn't block process exit
            thread = threading.Thread(target=thread_target, daemon=True)
            self._thread = thread
            thread.start()

            if is_async:

                def wait(self):
                    if not self._thread:
                        raise RuntimeError("No async execution in progress")
                    try:
                        self._thread.join()
                        if self._thread_exception:
                            raise self._thread_exception
                        return self._thread_result
                    except (KeyboardInterrupt, bdb.BdbQuit):
                        # Handle interrupts gracefully
                        logging.info("Execution interrupted")
                        return None

                wrapper.wait = wait  # type: ignore

            return wrapper

        return decorator

    return decorator


def with_clean_errors(func: Callable) -> Callable:
    """Decorator to add clean error handling to any function.

    This decorator wraps a function to catch any exceptions and format them
    using Plan's enhanced error reporting. It will print the clean error
    message and exit with a non-zero status code.

    Args:
        func: The function to wrap

    Returns:
        A wrapped function with clean error handling

    Example:
        ```python
        @with_clean_errors
        def main():
            model = AutoModel("meta-llama/Llama-3.2-1B", convert="megatron", setup=trainer)
            # ...

        if __name__ == "__main__":
            main()
        ```
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Format and print the error
            import sys

            formatted_error = Plan.format_exception(e)
            print(formatted_error, file=sys.stderr)

            # Exit with error code (this will terminate the program)
            sys.exit(1)

    return wrapper
