from collections import defaultdict

from nemo.common.plan.plan import Plan


class PlanRegistryMixin:
    _registry = defaultdict(dict)  # plan_type -> {name/pattern -> handler}

    @classmethod
    def register(cls, plan_type: str, *names_or_patterns: str):
        """Decorator to register a plan handler function.

        Args:
            plan_type: Type of plan to register (e.g., "importer", "setup")
            *names_or_patterns: One or more string identifiers or patterns

        Returns:
            Decorator function that registers the handler
        """

        def decorator(func):
            for name in names_or_patterns:
                cls._registry[plan_type][name] = func
            return func

        return decorator
    
    @classmethod
    def get_plan(
        cls, plan_type: str, plan_name: str, **kwargs
    ) -> Plan:
        """Get a plan by type and name.

        Args:
            plan_type: Type of plan to get (e.g., "importer", "setup")
            plan_name: Name or identifier of the plan
            **kwargs: Additional keyword arguments to pass to the plan constructor

        Returns:
            Instantiated plan object

        Raises:
            ValueError: If the plan type or name is not found in the registry
        """
        registry = cls._registry[plan_type]

        # First try exact match
        if plan_name in registry:
            return registry[plan_name](plan_name, **kwargs)

        # Then try pattern matching for setup-like registries
        import re

        for pattern, handler in registry.items():
            if "*" in pattern or "[" in pattern:
                # Convert our pattern syntax to regex
                regex_pattern = (
                    pattern.replace("*", ".*").replace("[", r"\[").replace("]", r"\]")
                )
                if re.match(regex_pattern, plan_name):
                    return handler(plan_name, **kwargs)

        raise ValueError(
            f"Unknown {plan_type} handler: {plan_name}. Available handlers: {list(registry.keys())}"
        )
