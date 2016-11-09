import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation
from keras.layers import Merge


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    from https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

class SiameseCNN:
    def __init__(self, nb_filters, filter_sizes, wordvector_size, max_words):
        self.nb_fitlers = nb_filters
        self.filter_sizes = filter_sizes

        self.node = self.create_node()
        self.max_words = max_words
        self.wordvector_size = wordvector_size

    def create_node(self):

        outputs = []

        for filter_length in self.filter_sizes:
            node = Sequential()
            node.add(Convolution1D(
                nb_filter=self.nb_fitlers,
                filter_length=filter_length
            ))
            node.add(Activation('relu'))
            node.add(MaxPooling1D())
            outputs.append(node)

        return Merge(outputs, mode='concat')

    def build_model(self):
        node = self.create_node()

        # Create two input nodes
        input_left = Input(shape=(self.max_words, self.wordvector_size))
        input_right = Input(shape=(self.max_words, self.wordvector_size))

        left = node(input_left)
        right = node(input_right)

        distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([left, right])

        model = Model(input=[left, right], output=eucl_dist_output_shape)

        return model