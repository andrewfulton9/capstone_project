from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD

import tempfile

import image_processing as ip
import aws_funcs as af



def image_model(nb_classes, img_size, weights_path):

    image_model = Sequential()

    image_model.add(Convolution2D(32, 3, 3,
                                  border_mode='valid',
                                  input_shape=(3, img_size, img_size)))
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Convolution2D(64, 3, 3,
                                  border_mode='valid'))
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Flatten())
    image_model.add(Dense(128))

    image_model.add(Dense(nb_classes))
    image_model.add(Activation('sigmoid'))

    image_model.compile(loss = 'binary_crossentropy',
                  optimizer = 'adadelta',
                  metrics = ['accuracy', 'recall']
                )

    return model

def vgg_16(weights_path=None, img_size=50):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, img_size, img_size)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

def vgg_19(weights_path=None, img_size=50):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, img_size, img_size)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

def save_weights(model, name):
    model.save_weights('weights/' + name + '_weights.h5')
    return

def fit_model(model, X_file, y_file, bucket = 'ajfcapstonearrays',
              weights_filename = 'VGG_16'):
    X, y = ip.get_Xy_data(X_file, y_file, bucket=bucket)
    model.fit(X, y)
    save_weights(model, weights_filename)

def get_y_filename(x_filename):
    ls_x_filename = x_filename.split('_')
    ls_x_filename[1] = 'y'
    y_filename = '_'.join(ls_x_filename)
    return y_filename

def fit_model_batches(X_filename, model=None, bucket = 'ajfcapstonearrays',
                      weights_filename='VGG_16_batch'):
    '''
    input: model = model to fit
           filename = name batches are saved under
           weights_filename = filename under which weights are saved
    output: new weights are saved
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
        model.train_on_batch(X,y)
        count += 1
    save_weights(model, weights_filename)
    X_test, y_test = ip.get_Xy_data(X_test, y_test, bucket = bucket)
    return X_test, y_test

def predict_model(img, model):
    probs = model.predict_proba(img)
    cats = model.predict_classes(img)
    return probs, cats

def evaluate_model(y_test, y_pred):
    pass



if __name__ == "__main__":
    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    # Test pretrained model
    # model = VGG_16('vgg16_weights.h5')
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # out = model.predict(im)
    # print np.argmax(out)
    pass
