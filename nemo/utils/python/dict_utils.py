import ast
from typing import Any, Dict


def update_nested_dict(dict_: Dict[str, Any], update: str, separator: str = '.') -> None:
    """
    Update a nested dictionary in-place with a key path and a new value. The update string should be in
    the format 'key1.key2.key3=new_value' for the default dot separator -- which hence must not be a part
    of any key. The new value will be converted to its appropriate type using ast.literal_eval.

    Args:
        dict_ (Dict[str, Any]): The dictionary to update.
        update (str): The update string in the format 'key1.key2.key3=new_value' (for separator=".").
        separator (str): The separator used to split the key path. Default is '.'.
    """
    # Split the update string into key path and new value
    assert update.count("=") == 1, "Update string must contain exactly one '=' to separate key path and value."
    key_path, value = update.split("=")
    keys = key_path.split(separator)

    # Traverse the nested dictionary & update the final key with new value
    current_dict = dict_
    for key in keys[:-1]:
        current_dict = current_dict[key]

    last_key = keys[-1]
    current_dict[last_key] = ast.literal_eval(value)
