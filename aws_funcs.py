import boto
import os
import tempfile

def aws_creds():
    key = os.environ['AWS_ACCESS_KEY_ID1']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']
    return key, secret_key

def connect_2_s3_bucket(bucket_name):
    key, secret_key = aws_creds()
    conn = boto.connect_s3(key, secret_key)
    b = conn.get_bucket(bucket_name)
    return b

def get_Xy_data(X_file, y_file, bucket = 'ajfcapstonearrays'):
    '''
    input: X_file = name of X data file in S3 bucket
           y_file = name of y data file in S3 bucket
           bucket = name of bucket to retrieve files from
    output: X and y arrays
    '''
    temp_dir = tempfile.mkdtemp()
    X_path = temp_dir + '/' + X_file + '.npy'
    y_path = temp_dir + '/' + y_file + '.npy'
    b = connect_2_s3_bucket(bucket)
    x_key = b.get_key(X_file)
    x_key.get_contents_to_filename(X_path)
    y_key = b.get_key(y_file)
    y_key.get_contents_to_filename(y_path)
    X = np.load(X_path)
    y = np.load(y_path)
    for f in [X_path, y_path]:
        os.remove(f)
    os.removedirs(temp_dir)
    return X, y
