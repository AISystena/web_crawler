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
import matplotlib
import matplotlib.pyplot as plt

import Util
from ImageCnn import ImageCnn
plt.style.use('ggplot')

"""
CNNによるテキスト分類 (posi-nega)
 - 5層のディープニューラルネット
 - 単語ベクトルにはWordEmbeddingモデルを使用
"""
class ImageTrainer:
    def __init__(self, gpu=0, epoch=15, batchsize=40):
        current_dir_path = os.path.dirname(__file__)
        self.model_pkl = current_dir_path + '/model/image_cnn.pkl'
        self.gpu         = gpu
        self.batchsize   = batchsize    # minibatch size
        self.n_epoch     = epoch        # エポック数(パラメータ更新回数)

        # 隠れ層のユニット数
        self.mid_units = 2560
        self.output_channel = 1280
        self.filters = 32
        self.n_label = 5
        self.input_channel = 3

    def dump_model(self, model):
        '''
        modelを保存
        '''
        with open(self.model_pkl, 'wb') as pkl:
            pickle.dump(model, pkl, -1)

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

    def train(self):
        # Prepare dataset
        dataset = Util.load_data()

        dataset['source'] = dataset['source'].astype(np.float32) #特徴量
        dataset['target'] = dataset['target'].astype(np.int32) #ラベル

        x_train, x_test, y_train, y_test = train_test_split(dataset['source'], dataset['target'], test_size=0.15)
        N_test = y_test.size         # test data size
        N = len(x_train)             # train data size

        print('input_channel is {}'.format(self.input_channel))
        print('output_channel is {}'.format(self.output_channel))
        print('filter_height is {}'.format(self.filters))
        print('n_label is {}'.format(self.n_label))

        #モデルの定義
        model = self.load_model()
        if model == None:
            model = L.Classifier(ImageCnn(self.input_channel, self.output_channel, self.filters, self.mid_units, self.n_label))

        xp = self.makeGpuAvailable(model)

        # Setup optimizer
        optimizer = optimizers.AdaGrad()
        optimizer.setup(model)

        train_loss = []
        train_acc  = []
        test_loss  = []
        test_acc   = []

        # Learning loop
        for epoch in six.moves.range(1, self.n_epoch + 1):

            print('epoch', epoch, '/', self.n_epoch)

            # training)
            perm = np.random.permutation(N) #ランダムな整数列リストを取得
            sum_train_loss     = 0.0
            sum_train_accuracy = 0.0
            for i in six.moves.range(0, N, self.batchsize):

                #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
                x = chainer.Variable(xp.asarray(x_train[perm[i:i + self.batchsize]])) #source
                t = chainer.Variable(xp.asarray(y_train[perm[i:i + self.batchsize]])) #target

                optimizer.update(model, x, t)

                sum_train_loss      += float(model.loss.data) * len(t.data)   # 平均誤差計算用
                sum_train_accuracy  += float(model.accuracy.data ) * len(t.data)   # 平均正解率計算用

            train_loss.append(sum_train_loss / N)
            train_acc.append(sum_train_accuracy / N)

            print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N)) #平均誤差

            # evaluation
            sum_test_loss     = 0.0
            sum_test_accuracy = 0.0
            for i in six.moves.range(0, N_test, self.batchsize):

                # all test data
                x = chainer.Variable(xp.asarray(x_test[i:i + self.batchsize]))
                t = chainer.Variable(xp.asarray(y_test[i:i + self.batchsize]))

                loss = model(x, t)

                sum_test_loss     += float(loss.data) * len(t.data)
                sum_test_accuracy += float(model.accuracy.data)  * len(t.data)

            test_loss.append(sum_test_loss / N_test)
            test_acc.append(sum_test_accuracy / N_test)
            print(' test mean loss={}, accuracy={}'.format(
                sum_test_loss / N_test, sum_test_accuracy / N_test)) #平均誤差

            if epoch > 10:
                optimizer.lr *= 0.97
                print('learning rate: ', optimizer.lr)

            sys.stdout.flush()

        # modelを保存
        self.dump_model(model)

        # 精度と誤差をグラフ描画
        plt.figure(figsize=(8,6))
        plt.plot(range(len(train_acc)), train_acc)
        plt.plot(range(len(test_acc)), test_acc)
        plt.legend(["train_acc","test_acc"],loc=4)
        plt.title("Accuracy of digit recognition.")
        plt.plot()
        plt.show()

if __name__ == '__main__':

    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  '    , '-g', dest='gpu'        , type=int, default=0,            help='0: use gpu, -1: use cpu')
    parser.add_argument('--data '    , '-d', dest='data'       , type=str, default='data',  help='an input data folder')
    parser.add_argument('--epoch'    , '-e', dest='epoch'      , type=int, default=10,          help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', dest='batchsize'  , type=int, default=40,           help='learning minibatch size')
    parser.add_argument('--nunits'   , '-n', dest='nunits'     , type=int, default=2000,          help='number of units')

    args = parser.parse_args()

    it = ImageTrainer()
    it.train()
