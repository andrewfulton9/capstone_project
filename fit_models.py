import CNN
import image_processing as ip
import score_models as sm
import aws_funcs as af
import numpy as np
import pandas as pd
import os


class EmployModel(object):

    def __init__(self, model, X_file,
                 arr_bucket = 'ajfcapstonearrays',
                 weight_bucket = 'ajfcapstoneweights',
                 weights_filename = None, img_size = 50,
                 lr = 0.001, epochs = 25, batch_size = 250):
        '''
        input: model = object of model being employed
               X_file = base filename of independant variable files
               arr_bucket = S3 bucket to retrieve data arrays from
               weight_bucket = S3 bucket to save weights into
               weights_filename = base name to save trained model weights into
               img_size = size of images being fed into model
               lr = learning rate of model
               epochs = # of epochs to use
               batch_size = size of batches in each epoch
        '''
        self.model = model
        self.X_file = X_file
        self.arr_bucket = arr_bucket
        self.weight_bucket = weight_bucket
        self.weights_filename = weights_filename
        self.img_size = img_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = self.model(img_size=self.img_size, lr = self.lr)

        self._b = af.connect_2_s3_bucket(self.arr_bucket)

        self.X_files = self.get_X_files()
        self.y_files = self.get_y_files()

        self.X_train = self.X_files[:-1]
        self.y_train = self.y_files[:-1]
        self.X_test, self.y_test = self.get_test_files()

        self.num_batches = len(self.X_train)

        self.fit_model_batches()

        self._bw = af.connect_2_s3_bucket(self.weight_bucket)
        self.save_weights_local()
        self.save_weights_remote()

    def fit_model_batches(self):
        '''
        input: none
        output: none

        fits a model iteratively through batches of data. For large training data
        sets
        '''
        for i, (X_train, y_train) in enumerate(zip(self.X_train,
                                                   self.y_train)):
            print 'fitting batch {} of {}:\n X = {}, y = {}'.format(\
                                i, self.num_batches, X_train, y_train)
            X, y = ip.get_Xy_data(X_train,y_train, bucket=self.arr_bucket)
            self.model.fit(X,y, nb_epoch = self.epochs,
                                batch_size = self.batch_size
                                )
        return

    def get_X_files(self):
        '''
        gets the names files for the independant variables (image arrays)
        '''
        x_files = [f.name for f in self._b.list() if self.X_file in f.name]
        return x_files

    def get_y_files(self):
        '''
        gets the names of files for the dependant variables (category arrays)
        '''
        y_files = [self.get_y_filename(f) for f in self.X_files]
        return y_files

    def get_test_files(self):
        '''
        gets files the model will be validated on
        '''
        X_test, y_test = ip.get_Xy_data(self.X_files[-1],
                                        self.y_files[-1],
                                        bucket = self.arr_bucket)
        y_test = self.reshape_y_test_arr(y_test)
        return X_test, y_test

    def get_y_filename(self, x_filename):
        '''
        INPUT: x_filename with X training/testing data
        OUTPUT: y_filename with y data corresponding to x_filename data
        '''
        ls_x_filename = x_filename.split('_')
        ls_x_filename[1] = 'y'
        y_filename = '_'.join(ls_x_filename)
        return y_filename

    def save_weights_local(self):
        '''
        save the weights from a fitted model to local weights folder
        '''
        self.model.save_weights('weights/' + self.weights_filename + '.h5')
        return

    def save_weights_remote(self):
        '''
        input: S3 bucket to save weights into
        output: None

        takes all the files from the weights folder and saves them in the S3 bucket
        '''
        for f in os.listdir('weights'):
            path = 'weights/' + f
            k = self._bw.new_key(f)
            k.set_contents_from_filename(path)
        return

    def reshape_y_test_arr(self, y_arr):
        '''
        input: y_arr = y test array shaped (n, 5)
        output: y test array reshaped to (n, 1)
        '''
        df = pd.DataFrame(y_arr)
        df[1] = df[1].apply(lambda x: 2 if x == 1 else 0)
        df[2] = df[2].apply(lambda x: 3 if x == 1 else 0)
        df[3] = df[3].apply(lambda x: 4 if x == 1 else 0)
        df[4] = df[4].apply(lambda x: 5 if x == 1 else 0)
        arr = df.sum(axis = 1).astype(int).values
        return arr - 1

    def test_probabilities(self):
        '''
        returns the probability of belonging to any of the classes for each
        image in the validation set
        '''
        return self.model.predict_proba(self.X_test)

    def test_classifications(self):
        '''
        returns the most likely classification for each image in the validation
        set
        '''
        return self.model.predict_classes(self.X_test)

    def accuracy(self):
        '''
        returns the accuracy of the model based on the predicted classes of the
        validation set vs. the real values of the validation set
        '''
        pred_classes = self.test_classifications()
        correct = sum([1 for a, b in zip(self.y_test, pred_classes) if a == b])
        return float(correct) / len(self.y_test)


if __name__ == '__main__':
    # build model to fit
    model = CNN.vgg_basic

    cnn = EmployModel(model, 'arr_X_50_full', lr = .0005, batch_size=999,
                      weights_filename = '50_full_basic_batchfit_3')

    print 'accuracy: ', cnn.accuracy()
