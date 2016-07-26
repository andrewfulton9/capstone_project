import boto
from boto.s3.key import Key
import skimage
from skimage.viewer import ImageViewer
from skimage.transform import resize
import numpy as np
import tempfile
import os
import sys

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

def get_img_array(bucket):
    access_key = os.environ['AWS_ACCESS_KEY_ID1']
    access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']
    conn = boto.connect_s3(access_key, access_secret_key)

    b = conn.get_bucket(bucket)
    files = [f.name for f in b.list()]

    ls = []
    temp_dir = tempfile.mkdtemp()

    for i, f in enumerate(files):
        if i % 1000 == 0:
            print '{}-{}: processing...'.format(i, i+1000)
        k = b.get_key(f)
        fn = temp_dir + '/' + str(f)
        k.get_contents_to_filename(fn)
        try:
            img = skimage.io.imread(fn)
        except:
            continue
        if img.shape[0] > 50:
            resized = resize(img, (50,50, 3))
            ls.append(np.transpose(resized))
        os.remove(fn)
        if i % 10000 == 0 and i > 0:
            arr = np.array(ls)
            fn = 'data_arrays/' + bucket + \
                 '_{}'.format(i) + '.npy'
            np.save(fn, arr)
            ls = []
    os.removedirs(temp_dir)
    arr = np.arr(ls)
    fn = '../data_arrays/' + bucket + \
         '_{}'.format(len(files)) + '.npy'
    np.save(fn, arr)
    return

if __name__ == '__main__':
    bucket = sys.argv[1]
    ls = get_img_array(bucket)
