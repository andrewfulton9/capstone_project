import CNN
import image_processing as ip
import score_models as sm
import aws_funcs as af
import numpy as np
import os


def save_weights(bucket = 'ajfcapstoneweights'):
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
    model = CNN.vgg_16(img_size=50)

    # fit models and return files to use for testing
    X_test, y_test = CNN.fit_model_batches('arr_X_50_full', model = model,
                                    weights_filename='50_full_VGG16_batchfit')

    # save weights to s3 bucket
    save_weights()

    # get the probability of each classification or each test observation, and
    # get the classification each observation was classified as
    probs, cats = CNN.predict_model(X_test, model)

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
