"""S3 utilities."""
from collections import namedtuple

from botocore.exceptions import ClientError

S3_PREFIX = "s3://"

ParsedS3Path = namedtuple("ParsedS3Path", ["bucket", "key"])


def is_s3_path(path: str) -> str:
    return path.startswith(S3_PREFIX)


def parse_s3_path(path: str) -> ParsedS3Path:
    assert is_s3_path(path)
    parts = path.replace(S3_PREFIX, "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    assert S3_PREFIX + bucket + "/" + key == path
    return ParsedS3Path(bucket=bucket, key=key)


def object_exists(client, path: str) -> bool:
    parsed_s3_path = parse_s3_path(path)
    try:
        response = client.head_object(Bucket=parsed_s3_path.bucket, Key=parsed_s3_path.key)
        return True
    except ClientError:
        return False
