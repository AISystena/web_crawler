# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class ImageCnn(Chain):

    def __init__(self, input_channel, output_channel, filters, mid_units, n_label):
        super(ImageCnn, self).__init__(
            # input_channel: 1:白黒 3:RGB など
            conv1 = L.Convolution2D(input_channel, output_channel, filters, pad=4, stride=4),
            #conv2 = L.Convolution2D(640, 640, 16, pad=2),
            #conv3 = L.Convolution2D(mid_units, output_channel, (filter_height, filter_width)),
            #conv4 = L.Convolution2D(mid_units, output_channel, (filter_height, filter_width)),
            #conv5 = L.Convolution2D(mid_units, output_channel, (filter_height, filter_width)),
            #conv6 = L.Convolution2D(mid_units, output_channel, (filter_height, filter_width)),
            l1    = L.Linear(output_channel, mid_units),
            l2    = L.Linear(mid_units,  n_label),
        )

    #Classifier によって呼ばれる
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        #h1 = F.dropout(F.relu(self.conv1(x)))
        #h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        #h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 3, stride=2)
        #h3 = F.dropout(F.relu(self.conv3(h2)))
        #h4 = F.max_pooling_2d(F.relu(self.conv4(h3)), 3)
        #h5 = F.dropout(F.relu(self.conv5(h4)))
        #h6 = F.max_pooling_2d(F.relu(self.conv6(h5)), 3)
        #h7 = F.dropout(F.relu(self.l1(h6)))
        h7 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h7)
        return y
