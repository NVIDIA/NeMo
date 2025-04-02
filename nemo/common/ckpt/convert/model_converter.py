from typing import TypeVar
from torch import nn
from nemo.common.plan.plan import Plan

ModelT = TypeVar("ModelT", bound=nn.Module)


class ModelConverter(Plan):
    _context_converters = {}
    _state_converters = {}
    _converting = set()
    name: str

    def __init__(self, output_path=None, overwrite: bool = False):
        super().__init__()        
        self.output_path = output_path
        self.overwrite = overwrite
    
    @classmethod
    def context_converter(cls, source: type, target: type):
        """Decorator to register a model importer function.

        Args:
            source: The source model class (e.g. LlamaForCausalLM)
            target: The target model class (e.g. LlamaModel)

        Returns:
            Decorator function that registers the importer
        """
        def decorator(func):
            if not isinstance(source, type):
                raise ValueError(f"Source must be a class, got {source}")

            # Register the importer with the class reference
            cls._context_converters[source] = (target, func)
            return func

        return decorator
    
    @classmethod
    def state_converter(cls, source: type, target: type):
        def decorator(func):
            if not isinstance(source, type):
                raise ValueError(f"Source must be a class, got {source}")

            # Register the importer with the class reference
            cls._state_converters[source] = (target, func)
            return func

        return decorator

    @classmethod
    def get_context_converter(cls, checkpoint_type: str, context):
        """Find the appropriate importer for the given context.

        Args:
            checkpoint_type: The type of checkpoint (e.g. "transformers", "nemo")
            context: The loaded checkpoint context

        Returns:
            tuple: (source_class, target_class, importer_func)

        Raises:
            ValueError: If no compatible importer is found
        """
        # Try to get model path from context if supported
        if hasattr(context, "model_path"):
            model_paths = context.model_path()
            if isinstance(model_paths, str):
                model_paths = [model_paths]

            errors = []
            for model_class in model_paths:
                # Look for matching source class
                for source_class, (target_class, importer) in cls._context_converters.items():
                    if source_class.__name__ == model_class:
                        return source_class, target_class, importer

            # If no exact match found, collect available source classes
            available_sources = [source.__name__ for source in cls._context_converters.keys()]
            errors.append(
                f"Model paths {model_paths} found but no exact match. "
                f"Available source classes: {available_sources}"
            )

            raise ValueError(
                f"Could not import model. Tried model paths: {model_paths}\n"
                f"Errors encountered:\n"
                + "\n".join(f"- {err}" for err in errors)
                + "\n"
                f"Available source classes: {[source.__name__ for source in cls._context_converters.keys()]}"
            )

        # Fallback to first available importer for the checkpoint type
        for source_class, (target_class, importer) in cls._context_converters.items():
            if source_class.__module__.startswith(checkpoint_type):
                return source_class, target_class, importer

        raise ValueError(
            f"No importers found for checkpoint type {checkpoint_type}. "
            f"Available source classes: {[source.__name__ for source in cls._context_converters.keys()]}"
        )

    @classmethod
    def get_state_converter(cls, checkpoint_type: str, context):
        """Find the appropriate state converter for the given context.

        Args:
            checkpoint_type: The type of checkpoint (e.g. "transformers", "nemo")
            context: The loaded checkpoint context

        Returns:
            converter: Plan

        Raises:
            ValueError: If no compatible converter is found
        """
        # Try to get model path from context if supported
        if hasattr(context, "model_path"):
            model_paths = context.model_path()
            if isinstance(model_paths, str):
                model_paths = [model_paths]

            errors = []
            for model_class in model_paths:
                # Look for matching source class
                for source_class, (target_class, converter) in cls._state_converters.items():
                    if source_class.__name__ == model_class:
                        return converter

            # If no exact match found, collect available source classes
            available_sources = [source.__name__ for source in cls._state_converters.keys()]
            errors.append(
                f"Model paths {model_paths} found but no exact match. "
                f"Available source classes: {available_sources}"
            )

            raise ValueError(
                f"Could not find state converter. Tried model paths: {model_paths}\n"
                f"Errors encountered:\n"
                + "\n".join(f"- {err}" for err in errors)
                + "\n"
                f"Available source classes: {[source.__name__ for source in cls._state_converters.keys()]}"
            )

        # Fallback to first available converter for the checkpoint type
        for source_class, (target_class, converter) in cls._state_converters.items():
            if source_class.__module__.startswith(checkpoint_type):
                return converter

        raise ValueError(
            f"No state converters found for checkpoint type {checkpoint_type}. "
            f"Available source classes: {[source.__name__ for source in cls._state_converters.keys()]}"
        )

    # def __call__(self, checkpoint_type: str, context):
    #     """Execute the model conversion using the appropriate importer.

    #     Args:
    #         checkpoint_type: The type of checkpoint (e.g. "transformers", "nemo")
    #         context: The loaded checkpoint context

    #     Returns:
    #         The imported model
    #     """
    #     _, _, importer = self.get_context_converter(checkpoint_type, context)
    #     return importer(context)
