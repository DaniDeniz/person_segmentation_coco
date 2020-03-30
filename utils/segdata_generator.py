import numpy as np
import cv2
import random
import sys

sys.path.append('../')

import math

import keras as keras




class DataGenerator(keras.utils.Sequence):
    """Abstract class that generates data for Keras. Method __data_generation needs to be redefines"""

    def __init__(self, list_ids, root_path, batch_size, n_classes, input_height, input_width, train=True):
        'Initialization'
        super(DataGenerator, self).__init__()
        self.batch_size = batch_size
        self.list_IDs = list_ids
        self.n_classes = n_classes
        self.root_path = root_path
        self.input_height = input_height
        self.input_width = input_width
        self.shuffle = train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return math.ceil(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


    def data_generation(self, list_ids_temp):
        x, y = get_batch(list_ids_temp, self.root_path,
                         self.n_classes, self.input_height,
                         self.input_width)
        return np.array(x), np.array(y)


def preprocess(image, height=224, width=224):
    #image = brightness_fix(image)
    im = np.zeros((height, width, 3), dtype='uint8')
    im[:, :, :] = 128

    if image.shape[0] >= image.shape[1]:
        scale = image.shape[0] / height
        new_width = int(image.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(image, (new_width, height))

        im[:, diff:diff + new_width, :] = img
    else:
        scale = image.shape[1] / width
        new_height = int(image.shape[0] / scale)
        diff = (height - new_height) // 2
        img = cv2.resize(image, (width, new_height))
        im[diff:diff + new_height, :, :] = img

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_norm = scale_frame(im)
    return im_norm, im


def brightness_fix(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_RGB = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
    img_YCrCb = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_YCrCb[:, :, 0])

    img_YCrCb[:, :, 0] = cl1

    img_RGB_2 = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2RGB)

    return img_RGB_2

def preprocess_with_label(image, label, height, width, n_classes):
    #image = brightness_fix(image)
    im = np.zeros((height, width, 3), dtype='uint8')
    im[:, :, :] = 128
    lim = np.zeros((height, width), dtype='uint8')

    if image.shape[0] >= image.shape[1]:
        scale = image.shape[0] / height
        new_width = int(image.shape[1] / scale)
        diff = (width - new_width) // 2
        img = cv2.resize(image, (new_width, height))
        label_img = cv2.resize(label, (new_width, height))

        im[:, diff:diff + new_width, :] = img
        lim[:, diff:diff + new_width] = label_img
    else:
        scale = image.shape[1] / width
        new_height = int(image.shape[0] / scale)
        diff = (height - new_height) // 2
        img = cv2.resize(image, (width, new_height))
        label_img = cv2.resize(label, (width, new_height))
        im[diff:diff + new_height, :, :] = img
        lim[diff:diff + new_height, :] = label_img
    lim = lim[:, :]
    seg_labels = np.zeros((height, width, n_classes))
    for c in range(n_classes):
        seg_labels[:, :, c] = (lim == (c+1)).astype(int)
    # seg_labels = np.reshape(seg_labels, (width * height, n_classes))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = scale_frame(im)
    return im, seg_labels

def get_batch(items, root_path, nClasses, height, width):
    x = []
    y = []
    for item in items:
        image_path = root_path + item.split(' ')[0]
        label_path = root_path + item.split(' ')[-1].strip()
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 0)
        im, seg_labels = preprocess_with_label(img, label_img, height, width, nClasses)
        x.append(im)
        y.append(seg_labels)
    return x, y

def scale_frame(x):
    x = x.astype(np.float32)
    x /= 127.5
    x -= 1.
    return x

def generator(root_path, path_file, batch_size, n_classes, input_height, input_width, train=True):
    f = open(path_file, 'r')
    items = f.readlines()
    f.close()
    if train:
        while True:
            shuffled_items = []
            index = [n for n in range(len(items))]
            random.shuffle(index)
            for i in range(len(items)):
                shuffled_items.append(items[index[i]])
            for j in range(len(items) // batch_size):
                x, y = get_batch(shuffled_items[j * batch_size:(j + 1) * batch_size],
                                 root_path, n_classes, input_height, input_width)
                yield np.array(x), np.array(y)
    else:
        for j in range(len(items) // batch_size):
            x, y = get_batch(items[j * batch_size:(j + 1) * batch_size],
                             root_path, n_classes, input_height, input_width)
            yield np.array(x), np.array(y)
