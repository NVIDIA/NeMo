import utils


class TestConvertFileNumbers:

    def test_convert_file_numbers(self):
        file_numbers = "0-3,6,10-16,23"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [0,1,2,3,6,10,11,12,13,14,15,16,23], f"Output of convert_file_numbers should be [0,1,2,3,6,10,11,12,13,14,15,16,23], but it is {list_file_numbers}."

    def test_convert_file_numbers_empty(self):
        file_numbers = ""
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [], f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."

    def test_convert_file_numbers_one_item(self):
        file_numbers = "7"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [7], f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."

    def test_convert_file_numbers_comma(self):
        file_numbers = "7,11,12,13,21,29"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [7,11,12,13,21,29], f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."

    def test_convert_file_numbers_dash(self):
        file_numbers = "7-11"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [7,8,9,10,11], f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."
