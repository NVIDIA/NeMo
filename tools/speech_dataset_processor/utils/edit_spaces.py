def remove_extra_spaces(input_string):
    """
	Removes extra spaces in between words and at the start and end
	of the string.
	e.g. "abc  xyz   abc xyz" --> "abc xyz abc xyz"
	e.g. " abc xyz " --> "abc xyz"
	"""
    output_string = " ".join(input_string.split())
    return output_string


def add_start_end_spaces(input_string):
    """
	Adds spaces at the start and end of the input string.
	This is useful for when we specify we are looking for a particular
	word " <word> ". This will ensure we will find the word even
	if it is at the beginning or end of the utterances (ie. there will
	definitely be two spaces around the word).

	e.g. "abc xyz" --> " abc xyz "
	"""
    # ensure no extra spaces
    no_extra_spaces_string = remove_extra_spaces(input_string)
    output_string = f" {no_extra_spaces_string} "

    return output_string

