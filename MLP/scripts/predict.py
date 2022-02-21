#!/usr/bin/env python
# coding: utf-8

import os
import csv
import itertools
import glob
import numpy as np
from keras.models import load_model
import pyaudio  # 録音機能を使うためのライブラリ
import wave     # wavファイルを扱うためのライブラリ
import feature as feat


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
temp_dir_path = base_absolute_path + "temp"
output_wav_path = temp_dir_path + "/predict.wav"
max_mean_csv_path = data_dir_path + "/max_mean.csv"

record_seconds = 4.0
input_device_index = 0

# 基本情報の設定
format = pyaudio.paInt16 # 音声のフォーマット
channels = 1             # モノラル
rate = 16000            # サンプルレート
chunk = 2**11            # データ点数


def recording():

    audio = pyaudio.PyAudio() # pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        input_device_index=input_device_index, # 録音デバイスのインデックス番号
                        frames_per_buffer=chunk)

    #--------------録音開始---------------
    print("recording...")

    frames = []
    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording")
    #--------------録音終了---------------

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(output_wav_path, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()


if __name__ == "__main__":

    max_means = []
    with open(max_mean_csv_path) as f:

        reader = csv.reader(f, delimiter=",")
        for row in reader:
            max_means.append(row)

    max_means = list(itertools.chain.from_iterable(max_means))

    if not os.path.isdir(temp_dir_path):
        os.mkdir(temp_dir_path)

    # 学習済みのモデルを読み込む
    model_path = glob.glob(data_dir_path + "/*.h5")[0]

    model = load_model(model_path)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    recording()
    features = feat.get_feature(output_wav_path)

    X = np.array(features, dtype=np.float32)
    X = np.array([X])

    X /= float(max_means[0])
    X -= float(max_means[1])

    predictions = model.predict(X)
    emotion = feat.emotion_labels[list(predictions[0]).index(predictions[0].max())]
    print("\n{} 確信率: {}%".format(emotion, predictions[0].max() * 100))

    print("\nAll process completed...")
