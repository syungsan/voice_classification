#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
from keras.models import load_model
import pyaudio  # 録音機能を使うためのライブラリ
import wave     # wavファイルを扱うためのライブラリ
import scipy.stats as sp
import feature as feat
import train as tr


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
temp_dir_path = base_absolute_path + "temp"
best_models_save_dir_path = data_dir_path + "/best_models"
output_wav_path = temp_dir_path + "/predict.wav"

model_number = 1
model_name = tr.model_names[model_number]

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

    if not os.path.isdir(temp_dir_path):
        os.mkdir(temp_dir_path)

    # 学習済みのモデルを読み込む
    model_paths = glob.glob(best_models_save_dir_path + "/*.h5")
    model_path = [m for m in model_paths if model_name in m][0]

    model = load_model(model_path)

    recording()
    features = feat.get_edited_feature(output_wav_path)

    X = np.array(features, dtype=np.float32)
    X = np.array([X])

    # 時系列に変換
    # X = np.reshape(X, (X.shape[0], feat.time_series_division_number, feat.feature_max_length), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    # X = np.reshape(X, (X.shape[0], (feat.time_series_division_number * feat.feature_max_length)))

    # Z-Score関数による正規化
    X = sp.zscore(X, axis=1)

    predictions = model.predict(X)
    emotion = feat.emotion_labels[list(predictions[0]).index(predictions[0].max())]
    print("\n{} 確信率: {}%".format(emotion, predictions[0].max() * 100))

    print("\nAll process completed...")
