import boto
from boto.s3.key import Key
import skimage
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

def get_img_array(in_bucket, out_bucket):
    access_key = os.environ['AWS_ACCESS_KEY_ID1']
    access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY1']
    conn = boto.connect_s3(access_key, access_secret_key)

    b1 = conn.get_bucket(in_bucket)
    b2 = conn.get_bucket(out_bucket)

    files = [f.name for f in b.list()]

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
            img = skimage.io.imread(fn)
        except:
            continue
        if img.shape[0] > 50:
            resized = resize(img, (50,50, 3))
            ls.append(np.transpose(resized))
        os.remove(fn)
        if i % 10000 == 0 and i > 0:
            filename = in_bucket + '_{}'.format(i) + '.npy'
            fp = temp_dir2 + '/' + filename
            np.save(fp, np.array(ls))
            k_out = b2.new_key(filename)
            k_out.set_contents_from_filename(fp)
            os.remove(fp)

    filename = in_bucket + '_{}'.format(len(files)) + '.npy'
    fp = temp_dir2 + '/' + filename
    np.save(fp, np.array(ls))
    k = b2.new_key(filename)
    k.set_contents_from_filename(fp)
    os.removedirs(temp_dir2)
    os.removedirs(temp_dir)
    return

if __name__ == '__main__':
    input_bucket = sys.argv[1]
    output_bucket = 'ajfcapstonearrays'
    ls = get_img_array(input_bucket, output_bucket)
