#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import scipy.stats as sp
import os
import csv
import codecs
from itertools import chain


# 特徴量の次元数
# MFCC: 39, PITCH: 2, VOLUME: 2, TEMPO: 1, RAW_WAV: 1
FEATURE_MAX_LENGTH = 43

# MFCCの区間平均の分割数
TIME_SERIES_DIVISION_NUMBER = 90

# "SimpleRNN", "SimpleRNNStack", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "BiLSTMStack", "BiGRUStack", "CNN_RNN_BiGRU", "CNN_RNN_BiLSTM"
MODEL_NAME = "CNN_RNN_BiLSTM"

MODEL_FILE_NAME = "CNN_RNN_BiLSTM_2020-03-09_final06_99.8%_model.h5"

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"
EMOTION_LIST_PATH = DATA_DIR_PATH + "/emotions.csv"
MODEL_FILE_PATH = LOG_DIR_PATH + "/" + MODEL_NAME + "/" + MODEL_FILE_NAME


def load_csv(file_name, delimiter):

    lists = []
    file = codecs.open(file_name, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def load_data(file_name):
    # load your data using this function

    # CSVの制限を外す
    # csv.field_size_limit(sys.maxsize)

    data = []
    target = []

    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=",")

        for columns in reader:
            target.append(columns[0])

            # こいつが決め手か？！
            data.append(columns[1:])

    data = np.array(data, dtype=np.float32)

    # なぜか進数のエラーを返すので処理
    target10b = []
    for tar in target:
        target10b.append(int(float(tar)))

    target = np.array(target10b, dtype=np.int32)

    return data, target


if __name__ == "__main__":

    emotion_list = load_csv(file_name=EMOTION_LIST_PATH, delimiter="\n")
    emotions = list(chain.from_iterable(emotion_list))

    X, y = load_data(file_name=TEST_FILE_PATH)

    # NANを0に置換
    # X = np.nan_to_num(X, nan=0.0)

    emotion_index = y

    # ラベルを one-hot-encoding形式 に変換
    y = np_utils.to_categorical(y)

    # 時系列に変換
    X = np.reshape(X, (X.shape[0], TIME_SERIES_DIVISION_NUMBER, FEATURE_MAX_LENGTH), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X = np.reshape(X, (X.shape[0], (TIME_SERIES_DIVISION_NUMBER * FEATURE_MAX_LENGTH)))

    # Z-Score関数による正規化
    X = sp.stats.zscore(np.array(X), axis=1)

    # 学習済みのモデルを読み込む
    model = load_model(MODEL_FILE_PATH)

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    score = model.evaluate(X, y, verbose=0)
    print("Test loss :", score[0])
    print("Test accuracy :", score[1])

    predictions = model.predict(X)

    i = 0
    for prediction in predictions:

        predict_word = emotions[list(prediction).index(prediction.max())]
        reference_word = emotions[emotion_index[i]]

        if predict_word == reference_word:
            print(reference_word + " == " + predict_word + " 正答 ○" + " 確信率 : " + str(prediction.max() * 100) + "%")
        else:
            print(reference_word + " != " + predict_word + " 誤答 ×" + " 確信率 : " + str(prediction.max() * 100) + "%")

        i += 1

    print("\nAll process completed...")
