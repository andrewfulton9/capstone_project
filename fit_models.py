import CNN
import image_processing as ip
import score_models as sm
import aws_funcs as af
import numpy as np


def save_weights(bucket = 'ajfcapstoneweights'):
    b = af.connect_2_s3_bucket(bucket)
    for f in os.listdir('weights'):
        path = 'weights/' + f
        k = b.new_key(f)
        k.set_contents_from_filename(path)

if __name__ == '__main__':
    model = CNN.vgg_16(img_size=50)

    X_test, y_test = CNN.fit_model_batches('arr_X_50_full', model = model,
                                    weights_filename='50_full_vgg16_batchfit')

    save_weights()

    probs, cats = CNN.predict_model(X_test, model)

    np.save('50_full_vgg16_probs.npy', probs)
    np.save('50_full_vgg16_cats.npy', cats)

    b = af.connect_2_s3('ajfcapstonearrays')

    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    b.new_key('50_full_vgg16_X_test.npy')
    b.set_contents_from_filename('X_test.npy')

    b.new_key('500_full_vgg16_y_test.npy')
    b.set_contents_from_filename('y_test.npy')
