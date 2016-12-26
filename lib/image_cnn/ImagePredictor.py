# coding: utf-8
import six
import sys
import glob
import argparse
import os.path
import pickle

import numpy as np
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import chainer
import chainer.links as L
#from chainer import optimizers, cuda, serializers
from chainer import optimizers, serializers
import chainer.functions as F

import Util

"""
CNNによるテキスト分類 (posi-nega)
 - 5層のディープニューラルネット
 - 単語ベクトルにはWordEmbeddingモデルを使用
"""
class ImagePredictor:
    def __init__(self, gpu=0):
        current_dir_path = os.path.dirname(__file__)
        self.model_pkl = current_dir_path + '/model/image_cnn.pkl'
        self.gpu         = gpu

    def load_model(self):
        '''
        modelを読み込む
        '''
        model = None
        if os.path.exists(self.model_pkl):
            with open(self.model_pkl, 'rb') as pkl:
                model = pickle.load(pkl)
        return model

    def makeGpuAvailable(self, model):
        #GPUを使うかどうか
        if self.gpu >= 0:
            pass
            #cuda.check_cuda_available()
            #cuda.get_device(self.gpu).use()
            #model.to_gpu()
        #xp = np if self.gpu < 0 else cuda.cupy #self.gpu <= 0: use cpu, otherwise: use gpu
        xp = np
        return xp

    def predict(self, image_path):
        #モデルの定義
        model = self.load_model()
        if model == None:
            print("model is empty")
            exit()

        xp = self.makeGpuAvailable(model)
        x = Util.load_image(image_path)
        x = xp.asarray(x.reshape((1,)+x.shape))
        y = xp.argmax(model.predictor(x).data, axis=1)
        return y[0]

if __name__ == '__main__':

    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  '    , '-g', dest='gpu'        , type=int, default=0,            help='0: use gpu, -1: use cpu')
    parser.add_argument('--data '    , '-d', dest='data'       , type=str, default='data',  help='an input data folder')
    parser.add_argument('--epoch'    , '-e', dest='epoch'      , type=int, default=10,          help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', dest='batchsize'  , type=int, default=40,           help='learning minibatch size')
    parser.add_argument('--nunits'   , '-n', dest='nunits'     , type=int, default=2000,          help='number of units')

    args = parser.parse_args()

    image_path = '/home/nakntu/work/playground/python/machine_learning/chainer_image/chainer_car_expo/image_cnn/data/airplanes/image_0001.jpg'
    ip = ImagePredictor()
    answer = ip.predict(image_path)
    print(answer)
