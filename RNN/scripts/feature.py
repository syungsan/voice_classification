#!/usr/bin/env python
# coding: utf-8

from aubio import source, pvoc, mfcc
import numpy as np
import os
import csv
from statistics import mean
import glob
from itertools import chain
import scipy.stats as sp


# MFCCの区間平均の分割数
TIME_SERIES_DIVISION_NUMBER = 90

# Training Only = 1, Training & Test = 2
METHOD_PATTERN = 2

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_WAV_DIR_PATH = DATA_DIR_PATH + "/wavs/train"
TEST_WAV_DIR_PATH = DATA_DIR_PATH + "/wavs/test"
TRAINING_FILE_PATH = DATA_DIR_PATH + "/train.csv"
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"
EMOTION_LIST_PATH = DATA_DIR_PATH + "/emotion_list.csv"

IS_MFCC = True # 39
IS_PITCH = True # 2
IS_VOLUME = True # 2
IS_TEMPO = True # 1
IS_RAW_WAV = False # 1


def write_csv(path, list):

    try:
        # 書き込み UTF-8
        with open(path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerows(list)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


def get_feature(filename):

    from aubio import pitch, tempo

    samplerate = 16000

    # for Computing a Spectrum
    win_s = 512 # Window Size
    hop_s = int(win_s / 4) # Hop Size 次回やりなおし！！

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    if IS_MFCC:
        n_filters = 40 # must be 40 for mfcc
        n_coeffs = 13
        p = pvoc(win_s, hop_s)
        m = mfcc(win_s, n_filters, n_coeffs, samplerate)
        mfccs = np.zeros([n_coeffs,])

    if IS_PITCH:
        tolerance = 0.8
        pitch_o = pitch("default", win_s, hop_s, samplerate)
        pitch_o.set_unit("Hz")
        pitch_o.set_silence(-40)
        pitch_o.set_tolerance(tolerance)
        pitches = []
        # confidences = []

    if IS_VOLUME:
        volumes = []

    if IS_TEMPO:
        tempo_o = tempo("specdiff", win_s, hop_s, samplerate)
        beats = []

    if IS_RAW_WAV:
        raw_wavs = []

    frames_read = 0

    while True:

        samples, read = s()

        if IS_MFCC:
            spec = p(samples)
            mfcc_out = m(spec)
            mfccs = np.vstack((mfccs, mfcc_out))

        if IS_PITCH:
            pitch = pitch_o(samples)[0]
            # pitch = int(round(pitch))
            # confidence = pitch_o.get_confidence()
            # if confidence < 0.8: pitch = 0.
            # print("%f %f %f" % (frames_read / float(samplerate), pitch, confidence))
            pitches += [float(pitch)]
            # confidences += [confidence]

        if IS_VOLUME:
            # Compute the energy (volume) of the
            # current frame.
            volume = np.sum(samples ** 2) / len(samples)
            # Format the volume output so that at most
            # it has six decimal numbers.
            volume = "{:.6f}".format(volume)
            volumes += [float(volume)]

        if IS_TEMPO:
            is_beat = tempo_o(samples)
            beats.append(float(is_beat))
            # if is_tempo:
                # this_beat = tempo_o.get_last_s()
                # beats.append(float(this_beat))
                # if o.get_confidence() > .2 and len(beats) > 2.:
                # break

        if IS_RAW_WAV:
            raw_wavs.append(list(map(float, samples)))

        frames_read += read
        if read < hop_s: break

    all_features = []

    if IS_MFCC:

        deltas = np.diff(mfccs, axis=0)
        ddeltas = np.diff(deltas, axis=0)

        mfccs = np.delete(mfccs, -1, 0)
        append = np.zeros((1, 13))
        ddeltas = np.r_[ddeltas, append]

        mfccs = mfccs.transpose()
        deltas = deltas.transpose()
        ddeltas = ddeltas.transpose()

        features = np.r_[mfccs, deltas, ddeltas]
        features = sp.stats.zscore(np.array(features), axis=1)

        all_features += features.tolist()

    if IS_PITCH:

        deltas = np.diff(pitches, axis=0)
        deltas = np.append(deltas, 0)

        features = np.r_[[np.array(pitches)], [deltas]]
        features = sp.stats.zscore(np.array(features), axis=1)

        # NaN消し
        features[np.isnan(features)] = 0.0

        all_features += features.tolist()

    if IS_VOLUME:

        deltas = np.diff(volumes, axis=0)
        deltas = np.append(deltas, 0)

        features = np.r_[[np.array(volumes)], [deltas]]
        features = sp.stats.zscore(np.array(features), axis=1)

        all_features += features.tolist()

    if IS_TEMPO:

        features = sp.stats.zscore(np.array([beats]), axis=1)

        # NaN消し
        features[np.isnan(features)] = 0.0

        all_features += features.tolist()

    if IS_RAW_WAV:

        features = [list(chain.from_iterable(raw_wavs))]
        features = sp.stats.zscore(np.array(features), axis=1)
        all_features += features.tolist()

    return all_features, frames_read


def section_average(input, division_number):

    if len(input) < TIME_SERIES_DIVISION_NUMBER:

        print("０埋めで音の長さを規定値に伸長...")

        additions = [0.0] * (TIME_SERIES_DIVISION_NUMBER - len(input))
        input += additions

    size = int(len(input) // division_number)
    mod = int(len(input) % division_number)

    index_list = [size] * division_number
    if mod != 0:
        for i in range(mod):
            index_list[i] += 1

    section_averages = []
    i = 0

    for index in index_list:
        section_averages.append(mean(input[i: i + index]))
        i += index

    return section_averages


if __name__ == "__main__":

    # 訓練とテスト用のデータをまとめて作る
    for method in range(0, METHOD_PATTERN):

        if method == 0:
            learning_file_path = TRAINING_FILE_PATH
            wav_folder_path = TRAINING_WAV_DIR_PATH
        else:
            learning_file_path = TEST_FILE_PATH
            wav_folder_path = TEST_WAV_DIR_PATH

        emotion_list = []
        trains = []

        feature_len = []

        emotions = os.listdir(wav_folder_path)

        for emotion in emotions:

            emotion_list.append([emotion])
            wav_path = wav_folder_path + "/" + emotion

            label = emotions.index(emotion)

            # パス内の全ての"指定パス+ファイル名"と"指定パス+ディレクトリ名"を要素とするリストを返す
            wav_file_paths = glob.glob("%s/*.wav" % wav_path)

            for wav_file_path in wav_file_paths:

                input_features, frames = get_feature(wav_file_path)

                section_averages = []

                for input_feature in input_features:

                    # 特徴量の時系列次元をチェックする
                    feature_len.append(len(input_feature))
                    section_averages.append(section_average(input_feature, TIME_SERIES_DIVISION_NUMBER))

                features = np.insert(section_averages, 0, label)
                trains.append(features)

                print(emotion + " => " + os.path.basename(wav_file_path) + ": frames: " + str(frames) + " was completed...")

            print(emotion + " completed...")

        write_csv(learning_file_path, trains)
        write_csv(EMOTION_LIST_PATH, emotion_list)

        print("特徴量の時系列の最小次元 : " + str(min(feature_len)))
        print("特徴量の時系列の最大次元 : " + str(max(feature_len)))

    print("\nAll process completed...")
