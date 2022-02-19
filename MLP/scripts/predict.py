#!/usr/bin/env python
# coding: utf-8

import os
import codecs
import csv
import numpy as np
from itertools import chain
from keras.models import load_model
import scipy.stats as sp
import pyaudio  # 録音機能を使うためのライブラリ
import wave     # wavファイルを扱うためのライブラリ


# 特徴量の次元数
# MFCC: 39, PITCH: 2, VOLUME: 2, TEMPO: 1, RAW_WAV: 1
FEATURE_MAX_LENGTH = 43

# MFCCの区間平均の分割数
TIME_SERIES_DIVISION_NUMBER = 90

MODEL_NAME = "CNN_RNN_BiLSTM"

MODEL_FILE_NAME = "CNN_RNN_BiLSTM_2020-03-09_final06_99.8%_model.h5"

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"
TEMP_DIR_PATH = BASE_ABSOLUTE_PATH + "temp"
WAV_OUTPUT_PATH = TEMP_DIR_PATH + "/predict.wav" # 音声を保存するファイル名
EMOTION_LIST_PATH = DATA_DIR_PATH + "/emotions.csv"
# MODEL_FILE_PATH = LOG_DIR_PATH + "/" + MODEL_NAME + "/" + MODEL_FILE_NAME
MODEL_FILE_PATH = DATA_DIR_PATH + "/" + MODEL_FILE_NAME

RECORD_SECONDS = 4.0 # 録音する時間の長さ（秒）

iDeviceIndex = 0 # 録音デバイスのインデックス番号

# 基本情報の設定
FORMAT = pyaudio.paInt16 # 音声のフォーマット
CHANNELS = 1             # モノラル
RATE = 16000            # サンプルレート
CHUNK = 2**11            # データ点数


def load_csv(file_name, delimiter):

    lists = []
    file = codecs.open(file_name, "r", 'utf-8')

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def recording():

    audio = pyaudio.PyAudio() # pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index = iDeviceIndex, # 録音デバイスのインデックス番号
                        frames_per_buffer=CHUNK)

    #--------------録音開始---------------

    print("recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)


    print("finished recording")

    #--------------録音終了---------------

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAV_OUTPUT_PATH, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == "__main__":

    import feature as feat

    emotion_list = load_csv(file_name=EMOTION_LIST_PATH, delimiter="\n")
    emotions = list(chain.from_iterable(emotion_list))

    if not os.path.isdir(TEMP_DIR_PATH):
        os.mkdir(TEMP_DIR_PATH)

    # 学習済みのモデルを読み込む
    model = load_model(MODEL_FILE_PATH)

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    recording()

    input_features, frames = feat.get_feature(WAV_OUTPUT_PATH)

    section_averages = []

    for input_feature in input_features:
        section_averages.append(feat.section_average(input_feature, TIME_SERIES_DIVISION_NUMBER))

    X = np.array(section_averages, dtype=np.float32)

    X = np.array([X.flatten()])

    # NANを0に置換
    # X = np.nan_to_num(X, nan=0.0)

    # 時系列に変換
    X = np.reshape(X, (X.shape[0], TIME_SERIES_DIVISION_NUMBER, FEATURE_MAX_LENGTH), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X = np.reshape(X, (X.shape[0], (TIME_SERIES_DIVISION_NUMBER * FEATURE_MAX_LENGTH)))

    # Z-Score関数による正規化
    X = sp.stats.zscore(np.array(X), axis=1)

    predictions = model.predict(X)

    predict_emotion = emotions[list(predictions[0]).index(predictions[0].max())]

    print(predict_emotion + " 確信率 : " + str(predictions[0].max() * 100) + "%")

    print("\nAll process completed...")
