import boto
import numpy as np
import os
import tempfile
from aws_funcs import connect_2_s3_bucket
import pandas as pd


def concat_arrs(cat, bucket):
    temp_dir = tempfile.mkdtemp()
    files = [f.name for f in bucket.list() if cat in f.name]
    ls = []
    for f in files:
        print 'processing: {}'.format(f)
        fn = temp_dir + '/' + str(f)
        k = b.get_key(f)
        k.get_contents_to_filename(fn)
        arr = np.load(fn)
        ls.append(arr)
        os.remove(fn)
    ls = np.array(ls)
    combined = np.concatenate([x for x in ls], axis = 0)
    os.removedirs(temp_dir)
    return combined

def shuffle_data(arr_1, arr_2):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(arr_1)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(arr_2)
    return arr_1, arr_2

if __name__ == '__main__':

    b = connect_2_s3_bucket('ajfcapstonearrays')

    car = concat_arrs('car', b)
    home = concat_arrs('home', b)
    # spec_events = concat_arrs('specevents', b)
    # savings = concat_arrs('saving', b)
    # travel = concat_arrs('travel', b)

    ls = [car, home]#, spec_events, savings, travel]

    y_ls = []
    for i, arr in enumerate(ls):
        print 'adding arrs to y_list'
        sub_ls = [i for x in xrange(len(arr))]
        print 'made sub_ls'
        ls.append(sub_ls)
        print 'appended sub_ls'
    print 'made y_ls'
    y_ls = np.array(y_ls)
    y_ls = np.ravel(y_ls)


    print 'concatting X_train'
    X_train = np.concatenate([ls], axis=0)
    print 'converting y_train to dummies'
    y_train = pd.get_dummies(pd.Series(y_ls)).values


    print 'shuffling data'
    X_train, y_train = shuffle_data(X_train, y_train)

    temp_dir = tempfile.mkdtemp()
    for arr, name in zip([X_train, y_train], ['X_train', 'y_train']):
        print 'saving X_train and y_train'
        fn = temp_dir + '/' + name + '.npy'
        np.save(fn, arr)
        k = b.new_key(name + '.npy')
        k.set_contents_from_filename(fn)
        os.remove(fn)
    os.removedirs(temp_dir)
