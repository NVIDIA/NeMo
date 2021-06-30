__all__ = ['is_url', 'has_numbers']

def is_url(input_str: str):
    """ Check if a string is a URL """
    url_segments = ['www', 'http', '.org', '.com', '.tv']
    return any(segment in input_str for segment in url_segments)

def has_numbers(input_str: str):
    """ Check if a string has a number character """
    return any(char.isdigit() for char in input_str)
