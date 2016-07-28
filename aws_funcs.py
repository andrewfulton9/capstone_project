import boto
import os

def aws_creds():
    key = os.environ['AWS_ACCESS_KEY_ID1']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']
    return key, secret_key

def connect_2_s3_bucket(bucket_name):
    key, secret_key = aws_creds()
    conn = boto.connect_s3(key, secret_key)
    b = conn.get_bucket(bucket_name)
    return b
