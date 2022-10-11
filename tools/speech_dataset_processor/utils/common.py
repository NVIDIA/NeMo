import os
import wget
import urllib
import tarfile

from nemo.utils import logging


# TODO: seems like this download code is saving initial file
#     in the local directory!


def download_file(source_url: str, target_directory: str):
    logging.info(
        f"Trying to download data from {source_url} "
        f"and save it in this directory: {target_directory}"
    )
    filename = os.path.basename(urllib.parse.urlparse(source_url).path)
    target_filepath = os.path.join(target_directory, filename)

    if os.path.exists(target_filepath):
        logging.info(
            f"Found file {target_filepath} => will not be attempting" f" download from {source_url}"
        )
    else:
        wget.download(source_url, target_directory)
        logging.info("Download completed")


def extract_archive(archive_path: str, extract_path: str) -> str:
    logging.info(
        f"Attempting to extract all contents from tar file {archive_path}"
        f" and save in {extract_path}"
    )
    archive_file_basename = os.path.basename(archive_path)
    archive_contents_dir = os.path.join(extract_path, archive_file_basename.split(".tar.gz")[0])

    if os.path.exists(archive_contents_dir):
        logging.info(
            f"Directory {archive_contents_dir} already exists => "
            "will not attempt to extract file"
        )
    else:
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(path=extract_path)
        logging.info("Finished extracting")

    return archive_contents_dir
