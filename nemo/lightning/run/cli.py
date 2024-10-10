def cli_entrypoint(*args0, **kwargs0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                import nemo_run as run

                return run.cli.entrypoint(*args0, **kwargs0)(func, *args, **kwargs)
            except (ImportError, ModuleNotFoundError) as error:
                logging.warning(f"Failed to import nemo.collections.llm.[api,recipes]: {error}")
                return func(*args, **kwargs)

        return wrapper

    return decorator
