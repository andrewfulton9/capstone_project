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

class ImageProcessing(object):

    def __init__(self, bucket_ls = ['ajfcapstonecars', 'ajfcapstonehome',
                                    'ajfcapstonesavings',
                                    'ajfcapstonespecevents',
                                    'ajfcapstonetravel'],
                 img_size = 50, sample_size = None, bin_arrays = True,
                 bin_size=1000, save_bucket= 'ajfcapstonearrays',
                 name = 'arr'):
        self.bucket_ls = bucket_ls
        self.img_size = img_size
        self.sample_size = sample_size
        self.bin_size = bin_size
        self.save_bucket = save_bucket
        self.name = name,

        print 'connecting to S3 buckets'
        self.bucket_dict = self.make_bucket_dict()
        print 'getting url dict'
        self.url_dict = self.get_url_dict()
        print 'getting url dataframe'
        self.url_df = self.url_dict_2_df()

        if sample_size == None:
            self.sample_str = 'full'
            self.sample_size = len(url_df.index)
        elif sample_size == 'half':
            self.sample_str = sample_size
            self.sample_size = len(self.url_df.index)/2
        else:
            self.sample_str = str(sample_size)
            self.sample_size = sample_size

        print 'sampling/shuffling df'
        self.sample_df = self.sample_df()

        if bin_arrays == True:
            self.bin_save_arrs()
        else:
            self.process_imgs()

    def make_bucket_dict(self):
        '''
        input: self
        output: dictionary where keys are bucket names and values are the objects
                connecting to the buckets
        '''
        b_dict = {}
        for bucket in self.bucket_ls:
            b_dict[bucket] = af.connect_2_s3_bucket(bucket)
        return b_dict


    def get_url_dict(self):
        '''
        input: self
        output: dictionary where key is name of s2 bucket and value is list of
                urls within the bucket
        '''
        d = {}
        for k, v in self.bucket_dict.items()
            d[k] = [f.name for f in v.list()]
        return d

    def url_dict_2_df(self):
        '''
        input: self
        output: dataframe containing the data from the passed dictionary.
                2 columns (url, bucket)
        '''
        d = {}
        for key in self.get_url_dict:
            df = pd.DataFrame({'url': url_dict[key],
                               'bucket': [key for x in url_dict[key]]})
            d[key] = df
        full_df = pd.concat([d[key] for key in d], axis = 0, ignore_index = True)
        return full_df

    def sample_df(self):
        '''
        input: self
        output: a dataframe randomly sampled from input dataframe
        '''
        return self.url_df.sample(n=self.sample_size)

    def build_np_arrs(self, df):
        '''
        input: self,
               df = df from which to build arrays
        output: X(IVs), and y(target) arrays
        '''
        temp_dir = tempfile.mkdtemp()
        X = np.empty((len(df.index), 3,
                      self.img_size, self.img_size))
        fill = np.empty((3, self.img_size, self.img_size))
        c = 0
        ind_list = []
        for ind, i in enumerate(df.index.copy()):
            if ind % 1000 == 0:
                print 'downloading and transforming imgs {} - {}'.format(ind,
                                                                         ind + 999)
            url = df.ix[i]['url']
            path = temp_dir + '/' + url
            k = self.bucket_dict[df.ix[i]['bucket']].get_key(url)
            k.get_contents_to_filename(path)
            try:
                img = io.imread(path)
                if img.shape[0] > 50:
                    resized = np.transpose(resize(img, (self.img_size,
                                                        self.img_size, 3)))
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

    def save_arrs(self, arr, name):
        '''
        input: arr = array to save,
        output: none

        saves files to bucket as name given
        '''
        temp_dir = tempfile.mkdtemp()
        name = name + '.npy'
        fp = temp_dir + '/' + name
        np.save(fp, arr)
        b = af.connect_2_s3_bucket(self.save_bucket)
        k = b.new_key(name)
        k.set_contents_from_filename(fp)
        os.remove(fp)
        os.removedirs(temp_dir)
        return

    def bin_save_arrs(self):
        '''
        input: self
        output: None

        A function to batch arrays into different files so the files are not too
        large to send back and forth from S3 buckets.
        '''
        for x in xrange(0, len(self.sampled_df.index), self.bin_size):
            print 'building bin: {}'.format(x)
            bin_df = self.sampled_df.iloc[x:x+bin_size-1,:].copy()
            print 'building X, y arrays'
            X, y = self.build_np_arrs(bin_df)
            print 'saving X array'
            self.save_arrs(X,
                          (self.name + '_X_{}_{}_bin{}'.format(self.img_size,
                                                               self.sample_str,
                                                               x)))
            print 'saving y array'
            self.save_arrs(y,
                          (self.name + '_y_{}_{}_bin{}'.format(self.img_size,
                                                               self.sample_str,
                                                               x)))
        print 'complete'



    def process_imgs(self):
        '''
        input: self
        output: none

        saves processed image arrays to s3
        '''
        print 'building X, y arrays'
        X, y = self.build_np_arrs(self.sampled_df)
        print 'saving X array'
        self.save_arrs(X, (self.name + '_X_{}_{}'.format(self.img_size,
                                                         self.sample_str)))
        print 'saving y array'
        self.save_arrs(y, (self.name + '_y_{}_{}'.format(self.img_size,
                                                         self.sample_str)))

if __name__ == '__main__':
    process = ImageProcessing()

    #say what?
