# -*- coding: utf-8 -*-
import gzip
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import os
from PIL import Image

# 今回はローカルディレクトリに保存したMNISTデータを使う
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

# 画像データを読み出す関数定義
def load_img(file_name):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data

# ラベルデータを読み出す関数定義
def load_label(file_name):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
    return labels

def write_text(image_path, number):
    img = cv.imread(image_path)
    h, w, c = img.shape
    cv.putText(img, str(number), (10, h),
               cv.FONT_HERSHEY_PLAIN, 8,
               (0, 128, 255), 1, cv.LINE_AA)
    res_name = os.path.splitext(os.path.basename(image_path))[0]
    res_name = res_name + "_result.jpg"

    cv.imwrite(res_name, img)
    
    res_img = cv.imread(res_name)
    plt.imshow(res_img)
    plt.show()
    

if __name__ == "__main__":

    # 学習データセットを取得する
    x_train = load_img(key_file['train_img'])
    y_train = load_label(key_file['train_label'])

    # テストデータセットを取得する
    x_test = load_img(key_file['test_img'])
    y_test = load_label(key_file['test_label'])

    # 正規化(RGB値0-255 -> 0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # ニューラルネットワークの構築
    # 入力層の設定
    model = tf.keras.models.Sequential()
    # 入力層: 28x28
    model.add(tf.keras.layers.Input((28, 28)))
    # 1次配列に変換
    model.add(tf.keras.layers.Flatten())

    # 中間層の設定
    # 128個に全結合
    model.add(tf.keras.layers.Dense(128))
    # 中間層の活性化関数の設定
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    # 20%ドロップアウトさせる
    model.add(tf.keras.layers.Dropout(0.2))

    # 出力層の設定
    model.add(tf.keras.layers.Dense(10))
    # 出力層の活性化関数の設定
    model.add(tf.keras.layers.Activation(tf.keras.activations.softmax))

    # 学習
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    model.fit(x_train, y_train, epochs=5)


    # 手書き文字画像の保存先
    img_dir = "mydata/"

    # 判定したい画像ファイル名
    img_names = ['one.jpg',
                'two.jpg',
                'three.jpg',
                'four.jpg',
                'five.jpg',
                'six.jpg',
                'seven.jpg',
                'eight.jpg',
                'nine.jpg',
                'zero.jpg']

    for img_name in img_names:
        img_path = img_dir + img_name
        img = Image.open(img_path).convert('L')
        img.thumbnail((28, 28)) # 28*28に変換
        img = np.array(img) # numpy arrayに変換

        pred = model.predict(img[np.newaxis])
        res = np.argmax(pred)
        res_img = write_text(img_path, res)
        
        file_name = os.path.basename(img_name)
        print(file_name + ": " + str(res))
