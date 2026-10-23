import json
import logging

from minio import Minio

from processor.import_processor.base import setup_logging
from processor.import_processor.import_config import get_config

config = get_config()
setup_logging()

try:
    minio_client = Minio(
        endpoint=config.minio_endpoint,
        access_key=config.minio_access_key,
        secret_key=config.minio_secret_key,
        secure=False
    )


    found = minio_client.bucket_exists(config.minio_bucket)

    if not found:
        minio_client.make_bucket(config.minio_bucket)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{config.minio_bucket}/*",
            }
        ],
    }

    minio_client.set_bucket_policy(config.minio_bucket, json.dumps(policy))
except Exception as e:
    logging.error("minio连接失败")

def get_minio_client():
    return minio_client

if __name__ == '__main__':
    get_minio_client()