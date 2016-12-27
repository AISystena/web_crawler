# coding: utf-8
import os.path
import pickle

import numpy as np
from chainer import cuda
import chainer.functions as F

import Util

"""
CNNによる画像分類 (posi-nega)
 - 5層のディープニューラルネット
"""


class ImagePredictor:
    def __init__(self, gpu=0):
        current_dir_path = os.path.dirname(__file__)
        self.model_pkl = current_dir_path + '/model/image_cnn.pkl'
        self.gpu = gpu

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
        # GPUを使うかどうか
        if self.gpu >= 0:
            pass
            cuda.check_cuda_available()
            cuda.get_device(self.gpu).use()
            model.to_gpu()
        xp = np if self.gpu < 0 else cuda.cupy
        return xp

    def predict(self, image_path):
        # モデルの定義
        model = self.load_model()
        if model is None:
            print("model is empty")
            exit()

        xp = self.makeGpuAvailable(model)
        x = Util.load_image(image_path)
        x = xp.asarray(x.reshape((1,)+x.shape))
        pred_y = F.softmax(model.predictor(x).data).data
        for i, p in enumerate(pred_y[0]):
            print("[{0:02d}]:{1:.3f}%".format(i, float(p)))
        y = xp.argmax(pred_y[0])
        return y
