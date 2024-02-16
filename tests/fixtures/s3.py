"""Fixtures for interacting with a mock S3 instance and the local filesystem."""
import contextlib
import os
import shlex
import subprocess
import time

import boto3
import pytest
import requests
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

from nemo.utils.s3 import S3_PREFIX

_PORT = 5555
_BUCKET = "test-bucket"
_ENDPOINT_URI = f"http://127.0.0.1:{_PORT}/"


@contextlib.contextmanager
def _ensure_safe_environment_variables():
    saved_environ = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


@pytest.fixture(scope="session")
def s3_server():
    """Spin up a local S3 server.

    We start a new process to host the server once per test session
    and terminate the process at the end of the session.
    """
    with _ensure_safe_environment_variables():
        os.environ["AWS_ACCESS_KEY_ID"] = "foobar_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "foobar_secret"
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = ""
        os.environ["AWS_CONFIG_FILE"] = ""

        proc = subprocess.Popen(
            shlex.split(f"moto_server -p {_PORT}"),
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

        timeout = 8
        while True:
            try:
                r = requests.get(_ENDPOINT_URI)
                if r.ok:
                    break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
            assert timeout > 0, "Timed out waiting for moto server"

        yield

        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def s3_client(s3_server):
    """Return an S3 client connected to a mock instance containing a bucket.

    This fixture is created once per test function. It is deleted when the test
    function is finished. Deletion includes deleting the bucket.
    """
    client = boto3.client("s3", endpoint_url=_ENDPOINT_URI, region_name="us-east-1")
    client.create_bucket(Bucket=_BUCKET, ACL="public-read-write")

    try:
        yield client
    finally:
        bucket = boto3.resource("s3", endpoint_url=_ENDPOINT_URI, region_name="us-east-1").Bucket(_BUCKET)
        bucket.objects.all().delete()
        bucket.delete()


@pytest.fixture
def s3(s3_client):
    """Return an S3FileSystem connected to a mock instance containing a bucket."""
    fs = S3FileSystem(
        anon=True, use_listings_cache=False, client_kwargs={"endpoint_url": _ENDPOINT_URI, "region_name": "us-east-1"}
    )
    yield fs


@pytest.fixture
def s3_parent_dir():
    return S3_PREFIX + _BUCKET


@pytest.fixture
def local():
    return LocalFileSystem()


@pytest.fixture
def local_parent_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("tmp"))
