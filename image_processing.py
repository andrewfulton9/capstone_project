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

def get_url_dict(ls):
    '''
    input: list of S2 buckets
    output: dictionary where key is name of s2 bucket and value is list of
            urls within the bucket
    '''
    d = {}
    for cat in ls:
        b = af.connect_2_s3_bucket(cat)
        d[cat] = [f.name for f in b.list()]
    return d

def url_dict_2_df(url_dict):
    '''
    input: dictionary of S2 buckets and lists of image files
    output: dataframe containing all the data from the passed
            dictionary dictionary. 2 columns (url, bucket)
    '''
    d = {}
    for key in url_dict:
        df = pd.DataFrame({'url': url_dict[key],
                           'bucket': [key for x in url_dict[key]]})
        d[key] = df
    full_df = pd.concat([d[key] for key in d], axis = 0, ignore_index = True)
    return full_df

def sample_df(url_df, sample_size = None):
    '''
    input: df of filenames and their bucket
    output: randomly sampled dataframe from input dataframe
    '''
    if sample_size:
        sample_df = url_df.sample(n=sample_size)
    else:
        sample_df = url_df.sample(frac=1)
    return sample_df

def make_bucket_dict(buckets_list):
    '''
    input: list of buckets
    output: dictionary where keys are bucket names and values are the objects
            connecting to the buckets
    '''
    b_dict = {}
    for buck in buckets_list:
        b_dict[buck] = af.connect_2_s3_bucket(buck)
    return b_dict

def build_np_arrs(df, img_size=50):
    '''
    input: sampled df
           size to downsample images to
    output: X(IVs), and y(target) arrays
    '''
    buck_dict = make_bucket_dict(df['bucket'].unique())
    temp_dir = tempfile.mkdtemp()
    X = np.empty((len(df.index), 3, img_size, img_size))
    fill = np.empty((3,img_size,img_size))
    c = 0
    ind_list = []
    for ind, i in enumerate(df.index.copy()):
        if ind % 1000 == 0:
            print 'downloading and transforming imgs {} - {}'.format(ind,
                                                                     ind + 999)
        url = df.ix[i]['url']
        path = temp_dir + '/' + url
        k = buck_dict[df.ix[i]['bucket']].get_key(url)
        k.get_contents_to_filename(path)
        try:
            img = io.imread(path)
            if img.shape[0] > 50:
                resized = np.transpose(resize(img, (img_size,img_size, 3)))
            else:
                raise Exception('')
        except:
            df.drop(i, axis = 0, inplace = True)
            resized = fill
            ind_list.append(ind)
        X[ind,:,:,:] = resized
        os.remove(path)
    X = np.delete(X, ind_list, axis = 0)
    os.removedirs(temp_dir)
    y = pd.get_dummies(df['bucket']).values
    return X, y

def save_arrs(arr, bucket, name):
    '''
    input: array to save,
           bucket to save array to,
           name to save array as
    output: none, saves files to bucket as name given
    '''
    temp_dir = tempfile.mkdtemp()
    name = name + '.npy'
    fp = temp_dir + '/' + name
    np.save(fp, arr)
    b = af.connect_2_s3_bucket(bucket)
    k = b.new_key(name)
    k.set_contents_from_filename(fp)
    os.remove(fp)
    os.removedirs(temp_dir)
    return

def bin_save_arrs(bucket_ls, img_size = 50,
                  sample_size = None, bin_size=None,
                  save_bucket= 'ajfcapstonearrays', name = 'arr'):
    print 'getting url_dict'
    url_dict = get_url_dict(bucket_ls)
    print 'building url_df'
    url_df = url_dict_2_df(url_dict)
    sample_str = sample_size
    if sample_size == 'half':
        sample_size = len(url_df.index)/2
        sample_str = 'half'
    if sample_size == None:
        sample_size = len(url_df.index)
        sample_str = 'full'
    print 'sample_size: {}'.format(sample_size)
    print 'sampling/shuffling df'
    sampled_df = sample_df(url_df, sample_size)
    for x in xrange(0, len(sampled_df.index), bin_size):
        print 'building bin: {}'.format(x)
        bin_df = sampled_df.iloc[x:x+bin_size-1,:].copy()
        print 'building X, y arrays'
        X, y = build_np_arrs(bin_df, img_size = img_size)
        print 'saving X array'
        save_arrs(X, save_bucket,
                  (name + '_X_{}_{}_bin{}'.format(img_size, sample_str, x)))
        print 'saving y array'
        save_arrs(y, save_bucket,
                  (name + '_y_{}_{}_bin{}'.format(img_size, sample_str, x)))
    print 'complete'



def process_imgs(bucket_ls, img_size = 50, sample_size = None,
                 save_bucket= 'ajfcapstonearrays', name = 'arr'):
    '''
    input: list of buckets,
           img_size = size to downsample images to
           sample_size = size or sample to return
           save_bucket = bucket to save arrays into
           name = name to save files to will have
                  _(image_size)_(sample_size).npy appended to end of name
    output: none

    saves arrays to s2
    '''
    print 'getting url_dict'
    url_dict = get_url_dict(bucket_ls)
    print 'building url_df'
    url_df = url_dict_2_df(url_dict)
    sample_str = sample_size
    if sample_size == 'half':
        sample_size = len(url_df.index)/2
        sample_str = 'half'
    if sample_size == None:
        sample_size = len(url_df.index)
        sample_str = 'full'
    print 'sample_size: {}'.format(sample_size)
    print 'sampling/shuffling df'
    sampled_df = sample_df(url_df, sample_size)
    print 'building X, y arrays'
    X, y = build_np_arrs(sampled_df, img_size = img_size)
    print 'saving X array'
    save_arrs(X, save_bucket,
              (name + '_X_{}_{}'.format(img_size, sample_str)))
    print 'saving y array'
    save_arrs(y, save_bucket,
              (name + '_y_{}_{}'.format(img_size, sample_str)))

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
    b = af.connect_2_s3_bucket(bucket)
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

if __name__ == '__main__':
    # input_bucket = sys.argv[1]
    # output_bucket = 'ajfcapstonearrays'
    #
    # get_img_array(input_bucket, output_bucket)

    bucket_ls = ['ajfcapstonecars', 'ajfcapstonehome', 'ajfcapstonesavings',
                 'ajfcapstonespecevents', 'ajfcapstonetravel']
    #
    # process_imgs(bucket_ls, img_size=100)

    # process_imgs(bucket_ls, img_size=50, sample_size = 25, name =
    #              'test_testing')

    bin_save_arrs(bucket_ls, img_size = 50,
                  sample_size = None, bin_size=1000,
                  save_bucket= 'ajfcapstonearrays', name = 'arr')
