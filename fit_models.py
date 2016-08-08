import CNN
import image_processing as ip
import score_models as sm
import aws_funcs as af
import numpy as np
import os



def fit_model(model, X_file, y_file, bucket = 'ajfcapstonearrays',
              weights_filename = 'VGG_16'):
    '''
    INPUT:
        model = model to fit
        X_file = filename of file in S3 bucket to use for your X training data
        y_file = filename of file in S3 bucket to use for you y training data
        bucket = bucket to get X_file and y_file from
        weights_filename = filename to save weights under
    OUTPUT: None

    fits a model
    '''
    X, y = ip.get_Xy_data(X_file, y_file, bucket=bucket)
    model.fit(X, y)
    save_weights_local(model, weights_filename)

def get_y_filename(x_filename):
    '''
    INPUT: x_filename with X training/testing data
    OUTPUT: y_filename with y data corresponding to x_filename data
    '''
    ls_x_filename = x_filename.split('_')
    ls_x_filename[1] = 'y'
    y_filename = '_'.join(ls_x_filename)
    return y_filename

def fit_model_batches(X_filename, model=None, bucket = 'ajfcapstonearrays',
                      weights_filename='VGG_16_batch',
                      batch_size = 999):
    '''
    input: model = model to fit
           filename = name batches are saved under
           weights_filename = filename under which weights are saved
    output: None

    fits a model iteratively through batches of data. For large training data
    sets
    '''
    b = af.connect_2_s3_bucket(bucket)
    x_files = [f.name for f in b.list() if X_filename in f.name]
    y_files = [get_y_filename(f) for f in x_files]
    X_test = x_files[-1]
    y_test = y_files[-1]
    num_batches = len(x_files[:-1])
    count = 1
    for X_train, y_train in zip(x_files[:-1], y_files[:-1]):
        print 'fitting batch {} of {}:\n X = {}, y = {}'.format(\
                            count, num_batches, X_train, y_train)
        X,y = ip.get_Xy_data(X_train,y_train, bucket=bucket)
        model.fit(X,y, nb_epoch = 25, batch_size = batch_size)
        count += 1
    save_weights_local(model, weights_filename)
    X_test, y_test = ip.get_Xy_data(X_test, y_test, bucket = bucket)
    return X_test, y_test

def save_weights_local(model, name):
    '''
    INPUT: model = fitted model
           name = filename for weights to save
    OUTPUT: None

    save the weights from a fitted model to local weights folder
    '''
    model.save_weights('weights/' + name + '_weights.h5')
    return

def save_weights_remote(bucket = 'ajfcapstoneweights'):
    '''
    input: S3 bucket to save weights into
    output: None

    takes all the files from the weights folder and saves them in the S3 bucket
    '''
    b = af.connect_2_s3_bucket(bucket)
    for f in os.listdir('weights'):
        path = 'weights/' + f
        k = b.new_key(f)
        k.set_contents_from_filename(path)

if __name__ == '__main__':
    # build model to fit
    model = CNN.basic(img_size=100)

    # fit models and return files to use for testing
    X_test, y_test = fit_model_batches('arr_X_100_full', model = model,
                                    weights_filename='100_full_basic_batchfit')

    # save weights to s3 bucket
    save_weights_remote()

    # get the probability of each classification or each test observation, and
    # get the classification each observation was classified as
    probs, cats = sm.predict_model(X_test, model)

    # save the probability and category arrays
    np.save('50_full_VGG16_probs.npy', probs)
    np.save('50_full_VGG16_cats.npy', cats)

    # connect to S3 bucket
    b = af.connect_2_s3_bucket('ajfcapstonearrays')

    # save X_test, and y_test files locally
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # save X_test to S3 bucket
    k = b.new_key('50_full_VGG16_X_test.npy')
    k.set_contents_from_filename('X_test.npy')

    # save y_test to S3 bucket
    k = b.new_key('50_full_VGG16_y_test.npy')
    k.set_contents_from_filename('y_test.npy')
