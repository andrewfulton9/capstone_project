import boto
from boto.s3.key import Key
import skimage
from skimage.transform import resize
from skimage import io
import numpy as np
import tempfile
import os
import sys
import aws_funcs as af
import pandas as pd

'''
For each bucket of images
get all the urls for each image
instantiate and empty list
download each image
    get the array version of each image
    resize each array
    transpose the resized array
    add the transposed array to a list
    delete the image from the file
turn list into numpy array

'''

def get_img_array(in_bucket, out_bucket):
    access_key = os.environ['AWS_ACCESS_KEY_ID1']
    access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']
    conn = boto.connect_s3(access_key, access_secret_key)

    b1 = conn.get_bucket(in_bucket)
    b2 = conn.get_bucket(out_bucket)

    files = [f.name for f in b1.list()]

    ls = []
    temp_dir = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()

    for i, f in enumerate(files):
        if i % 1000 == 0:
            print '{}-{}: processing...'.format(i, i+1000)
        k = b1.get_key(f)
        fn = temp_dir + '/' + str(f)
        k.get_contents_to_filename(fn)
        try:
            img = io.imread(fn)
        except:
            print 'skipping: {}'.format(i)
            continue
        if img.shape[0] > 50:
            resized = resize(img, (50,50, 3))
            ls.append(np.transpose(resized))
        os.remove(fn)
        if i % 5000 == 0 and i > 0:
            filename = in_bucket + '_{}'.format(i) + '.npy'
            fp = temp_dir2 + '/' + filename
            np.save(fp, np.array(ls))
            print 'temp_dir: ',os.listdir(temp_dir2)
            k_out = b2.new_key(filename)
            k_out.set_contents_from_filename(fp)
            ls = []
            os.remove(fp)

    filename = in_bucket + '_{}'.format(len(files)) + '.npy'
    fp = temp_dir2 + '/' + filename
    np.save(fp, np.array(ls))
    k = b2.new_key(filename)
    k.set_contents_from_filename(fp)
    os.removedirs(temp_dir2)
    os.removedirs(temp_dir)
    return

def get_url_dict(ls):
    d = {}
    for cat in ls:
        b = af.connect_2_s3_bucket(cat)
        d[cat] = [f.name for f in b.list()]
    return d

def url_dict_2_df(url_dict):
    d = {}
    for key in url_dict:
        df = pd.DataFrame({'url': url_dict[key],
                           'bucket': [key for x in url_dict[key]]})
        d[key] = df
    full_df = pd.concat([d[key] for key in d], axis = 0, ignore_index = True)
    return full_df

def sample_df(url_df, sample_size):
    sample_df = url_df.sample(n=sample_size)
    return sample_df

def make_bucket_dict(buckets_list):
    b_dict = {}
    for buck in buckets_list:
        b_dict[buck] = af.connect_2_s3_bucket(buck)
    return b_dict

def build_np_arrs(df):
    buck_dict = make_bucket_dict(df['bucket'].unique())
    temp_dir = tempfile.mkdtemp()
    X = np.empty((len(df.index), 3, 50, 50))
    c = 0
    for ind, i in enumerate(df.index.copy()):
        url = df.ix[i]['url']
        path = temp_dir + '/' + url
        k = buck_dict[df.ix[i]['bucket']].get_key(url)
        k.get_contents_to_filename(path)
        try:
            img = io.imread(path)
        except:
            df.drop(i, axis = 0, inplace = True)
        if img.shape[0] > 50:
            resized = resize(img, (50,50, 3))
        else:
            df.drop(i, axis = 0, inplace = True)
        X[i,:,:,:] = np.transpose(resized)
        os.remove(path)
    os.removedirs(temp_dir)
    y = pd.get_dummies(df['bucket']).values
    return X, y

if __name__ == '__main__':
    # input_bucket = sys.argv[1]
    # output_bucket = 'ajfcapstonearrays'
    #
    # get_img_array(input_bucket, output_bucket)

    bucket_ls = ['ajfcapstonecars', 'ajfcapstonehome', 'ajfcapstonesavings',
                 'ajfcapstonespecevents', 'ajfcapstonetravel']

    url_dict = get_url_dict(bucket_ls)
    url_df = url_dict_2_df(url_dict)
    sampled_df = sample_df(url_df, 10)
    X, y = build_np_arrs(sampled_df)
