# coding: utf-8
import numpy as np
import os
import glob
from PIL import Image
from Categories import Categories

current_dir_path = os.path.dirname(__file__)
data_dir = current_dir_path + '/data'


def createDirLabel():
    dir_names = os.listdir(data_dir)
    id_list_file_name = './id_list.txt'

    with open(id_list_file_name, 'w') as id_list_file:
        for (i, dir_name) in enumerate(dir_names):
            print('{:03}:{}'.format(i, dir_name))
            id_list_file.write('{:03}:{}\n'.format(i, dir_name))
    return dir_names


def load_data():
    dir_names = createDirLabel()
    data_set = {}
    source_data_list = []
    target_data_list = []

    for (label, dir_name) in enumerate(dir_names):
        image_file_list = glob.glob(data_dir +'/'+ dir_name + '/*.jpg')
        for image_file in image_file_list:
            target_data_list.append(label)
            source_data_list.append(load_image(image_file))

    data_set['source'] = np.array(source_data_list)
    data_set['target'] = np.array(target_data_list)
    return data_set


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # image = Image.open(image_path).convert('L')
    image_w, image_h = _image_shape()
    w, h = image.size
    if w > h:
        shape = (int(image_w * w / h), int(image_h))
    else:
        shape = (int(image_w), int(image_h * h / w))
    x = (shape[0] - image_w) / 2
    y = (shape[1] - image_h) / 2

    # crop ...PIL(Image)の画像から一部を抜き出す
    pixels = np.asarray(image.resize(shape).crop((x, y, x + image_w, y + image_h))).astype(np.float32)
    # numpy配列の並びを[2,0,1]に並び替える
    # pixelsは3次元でそれぞれの軸は[Y座標, X座標, RGB]を表す
    # 入力データは4次元で[画像インデックス, BGR, Y座標, X座標]なので、配列の変換を行う
    # RGBからBGRに変換する
    pixels = pixels[:, :, ::-1].transpose(2, 0, 1)
    # 平均画像を引く
    # pixels -= _mean_image()
    pixels /= 255
    return pixels


def _image_shape():
    return (224, 224)
    # return (32, 32)


def _mean_image():
    # mean_image = np.ndarray((3, 224, 224), dtype=np.float32)
    mean_image = np.ndarray((3, 32, 32), dtype=np.float32)
    mean_image[0] = 103.939
    mean_image[1] = 116.779
    mean_image[2] = 123.68
    return mean_image
