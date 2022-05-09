import os
import json
import zstandard as zstd
import requests_mock

from bignlp.data_preparation.pile_dataprep_scripts import utils


FILE_NAME = "test_data.jsonl.zst"


def _create_json_zst():
    test_data = {"a": "b", "c": "d"}
    with open(FILE_NAME, "wb") as f:
        cctx = zstd.ZstdCompressor()
        with cctx.stream_writer(f) as compressor:
            compressor.write(json.dumps(test_data).encode())


def _read_json_zst():
    with open(FILE_NAME, "rb") as f:
        return f.read()


class TestConvertFileNumbers:
    def test_convert_file_numbers(self):
        file_numbers = "0-3,6,10-16,23"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [0, 1, 2, 3, 6, 10, 11, 12, 13, 14, 15, 16, 23], \
                f"Output of convert_file_numbers should be [0,1,2,3,6,10,11,12,13,14,15,16,23], but it is {list_file_numbers}."

    def test_convert_file_numbers_empty(self):
        file_numbers = ""
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert (
            list_file_numbers == []
        ), f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."

    def test_convert_file_numbers_one_item(self):
        file_numbers = "7"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [7], \
                f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."

    def test_convert_file_numbers_comma(self):
        file_numbers = "7,11,12,13,21,29"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [7, 11, 12, 13, 21, 29], \
                f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."

    def test_convert_file_numbers_dash(self):
        file_numbers = "7-11"
        list_file_numbers = utils.convert_file_numbers(file_numbers)
        assert list_file_numbers == [7, 8, 9, 10, 11], \
                f"Output of convert_file_numbers should be [] but it is {list_file_numbers}."


class TestSplitList:
    def test_split_list_empty(self):
        pass


class TestExtractSingleZSTFile:
    def test_mock_extract_single_zst_file(self):
        url = "http://test.com/file"
        save_dir = "."
        extracted_file_name = "extracted_data.jsonl"

        _create_json_zst()
        utils.extract_single_zst_file(FILE_NAME, save_dir, extracted_file_name)

        assert os.path.isfile(extracted_file_name)

        test_data = {"a": "b", "c": "d"}
        with open(extracted_file_name, "rb") as f:
            assert f.read() == json.dumps(test_data).encode()

        os.remove(FILE_NAME)
        os.remove(extracted_file_name)


class TestDownloadSingleFile:
    def test_mock_download_single_file(self):
        FILE_NAME = "test_data.jsonl.zst"

        url = "http://test.com/file"
        save_dir = "."
        downloaded_file_name = "downloaded_data.jsonl.zst"

        _create_json_zst()

        with requests_mock.Mocker() as mock:
            mock.head(requests_mock.ANY, headers={"content-length": 1000})
            mock.get(url, content=_read_json_zst(), headers={"content-length": "100"})
            utils.download_single_file(url, save_dir, downloaded_file_name)

        assert os.path.isfile(downloaded_file_name)

        with open(downloaded_file_name, "rb") as f:
            assert f.read() == _read_json_zst()

        os.remove(FILE_NAME)
        os.remove(downloaded_file_name)
