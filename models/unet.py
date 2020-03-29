from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Reshape, Permute, Activation, ZeroPadding2D, Cropping2D, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.initializers import orthogonal, constant, he_normal
from keras.regularizers import l2


def Unet(nClasses, input_height=256, input_width=256, nChannels=3):
    inputs = Input(shape=(input_height, input_width, nChannels))
    # encode
    # 224x224
    conv1 = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=orthogonal())(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same', activation="relu", kernel_initializer=orthogonal())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 112x112
    conv2 = Conv2D(128, (3, 3), padding='same', activation="relu", kernel_initializer=orthogonal())(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same', activation="relu",kernel_initializer=orthogonal())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 56x56
    conv3 = Conv2D(256, (3, 3), padding='same', activation="relu",kernel_initializer=orthogonal())(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same', activation="relu",kernel_initializer=orthogonal())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 28x28
    conv4 = Conv2D(512, (3, 3), padding='same', activation="relu", kernel_initializer=orthogonal())(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same', activation="relu", kernel_initializer=orthogonal())(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 14x14
    conv5 = Conv2D(1024, (3, 3), padding='same', activation="relu",kernel_initializer=orthogonal())(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same',activation="relu", kernel_initializer=orthogonal())(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # 8x8

    # decode
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv4], axis=-1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv3], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv2], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv1], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=orthogonal())(conv9)

    conv10 = Conv2D(nClasses, (1, 1), padding='same', activation='relu',
                    kernel_initializer=he_normal())(conv9)

    conv11 = (Reshape((input_height * input_width, -1)))(conv10)
    conv11 = (Activation('softmax'))(conv11)

    model = Model(input=inputs, output=conv11)

    return model
