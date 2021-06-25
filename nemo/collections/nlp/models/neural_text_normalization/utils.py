import os
import math
import torch
import random
import string
import numpy as np

from os.path import join
from nltk import word_tokenize

# Check if a string is a URL
def is_url(input_str):
    url_segments = ['www', 'http', '.org', '.com', '.tv']
    return any(segment in input_str for segment in url_segments)

# Check if a string has a number character
def has_numbers(input_str):
    return any(char.isdigit() for char in input_str)
