from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD



def image_model(nb_classes, input_shape = (3,50,50)):

    image_model = Sequential()

    image_model.add(Convolution2D(32, 3, 3,
                                  border_mode='valid',
                                  input_shape=input_shape))
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

def VGG_16(weights_path=None, in_shape = (3, 50, 50)):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=in_shape))
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

    else:
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')


    return model

if __name__ == "__main__":
    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print np.argmax(out)
