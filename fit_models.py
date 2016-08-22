import CNN
import image_processing as ip
import score_models as sm
import aws_funcs as af
import numpy as np
import os


class EmployModel(object):

    def __init__(model, X_file,
                 arr_bucket = 'ajfcapstonearrays',
                 weight_bucket = 'ajfcapstoneweights'
                 weights_filename = None, img_size = 50,
                 lr = 0.001, epochs = 25, batch_size = 250,):
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
        self._bw = af.connect_2_s3_bucket(self.weight_bucket)

        self.X_files = self.get_X_files()
        self.y_files = self.get_y_files()

        self.X_train = self.X_files[:-1]
        self.y_train = self.y_files[:-1]
        self.X_test, self.y_test = self.get_test_files()

        self.fit_model_batches()
        self.save_weights_local()
        self.save_weights_remote()

    def fit_model_batches():
        '''
        input: model = model to fit
               filename = name batches are saved under
               weights_filename = filename under which weights are saved
        output: None

        fits a model iteratively through batches of data. For large training data
        sets
        '''
        for i, (X_train, y_train) in enumerate(zip(self.X_train,
                                                   self.y_train)):
            print 'fitting batch {} of {}:\n X = {}, y = {}'.format(\
                                i, self.num_batches, X_train, y_train)
            X, y = ip.get_Xy_data(X_train,y_train, bucket=self.arr_bucket)
            self.model.fit(X,y, nb_epoch = self.epochs,
                                batch_size = self.batch_size)
        return

    def get_X_files():
        x_files = [f.name for f in self._b.list() if self.X_file in f.name]
        return x_files

    def get_y_files():
        y_files = [self.get_y_filename(f) for f in self.X_files]
        return y_files

    def get_test_files():
        X_test, y_test = ip.get_Xy_data(self.X_files[-1],
                                        self.y_files[-1],
                                        bucket = self.arr_bucket)
        return X_test, y_test

    def get_y_filename(x_filename):
        '''
        INPUT: x_filename with X training/testing data
        OUTPUT: y_filename with y data corresponding to x_filename data
        '''
        ls_x_filename = x_filename.split('_')
        ls_x_filename[1] = 'y'
        y_filename = '_'.join(ls_x_filename)
        return y_filename

    def save_weights_local():
        '''
        save the weights from a fitted model to local weights folder
        '''
        self.model.save_weights('weights/' + self.weights_filename + '.h5')
        return

    def save_weights_remote():
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

if __name__ == '__main__':
    # build model to fit
    model = CNN.vgg_basic

    cnn = EmployModel(model, 'arr_X_50_full')

    # get the probability of each classification or each test observation, and
    # get the classification each observation was classified as
    probs, cats = sm.predict_model(cnn.X_test, cnn.model)
