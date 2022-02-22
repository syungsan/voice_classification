#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import csv
import glob
import scipy.signal
from aubio import source, pvoc, mfcc, tempo, filterbank
from sklearn.model_selection import train_test_split
import scipy.stats as sp
from statistics import mean
# from sklearn import preprocessing

# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
raw_wav_dir_path = base_absolute_path + "wavs/raws"
augment_wav_dir_path = base_absolute_path + "wavs/augment"
data_dir_path = base_absolute_path + "data"
training_file_path = data_dir_path + "/train.csv"
test_file_path = data_dir_path + "/test.csv"

# 感情ラベルのリスト
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# MFCCの区間平均の分割数（1~）
time_series_division_number = 1

is_mfcc = True
is_pitch = True
is_volume = True
is_tempo = True

# 特徴量の次元数
feature_max_length = 0

if is_mfcc:
    feature_max_length += 39
if is_pitch:
    feature_max_length += 2
if is_volume:
    feature_max_length += 2
if is_tempo:
    feature_max_length += 1


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


# 高域強調
def preEmphasis(wave, p=0.97):

    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)


def get_feature(file_path):

    samplerate = 16000

    # for Computing a Spectrum
    win_s = 512 # Window Size
    hop_size = int(win_s / 4) # Hop Size

    all_features = []

    if is_mfcc:

        src = source(file_path, samplerate, hop_size)
        samplerate = src.samplerate

        _p = 0.97
        n_filters = 40 # must be 40 for mfcc
        n_coeffs = 13
        total_frames = 0
        total_samples = np.array([])

        pv = pvoc(win_s, hop_size)
        f = filterbank(n_filters, win_s)
        f.set_mel_coeffs_slaney(samplerate)
        energies = np.zeros((n_filters,))

        while True:

            hop_samples, read = src()  # read hop_size new samples from source
            total_samples = np.append(total_samples, hop_samples)

            fftgrain = pv(hop_samples)
            new_energies = f(fftgrain)
            energies = np.vstack([energies, new_energies])

            total_frames += read   # increment total number of frames
            if read < hop_size:    # end of file reached
                break

        # preEmphasis
        total_samples = preEmphasis(total_samples, _p).astype("float32")

        p = pvoc(win_s, hop_size)
        m = mfcc(win_s, n_filters, n_coeffs, samplerate)
        mfccs = np.zeros([n_coeffs,])
        index = 1

        while True:

            old_frame = hop_size * (index - 1)
            cur_frame = hop_size * index

            if total_frames - old_frame < hop_size:
                samples = total_samples[old_frame:total_frames]

                # ケツを0で埋めてhopサイズに間に合わせる
                samples = np.pad(samples, [0, hop_size - (total_frames - old_frame)], "constant")
            else:
                samples = total_samples[old_frame:cur_frame]

            spec = p(samples)
            mfcc_out = m(spec)
            mfccs = np.vstack((mfccs, mfcc_out))

            if total_frames - old_frame < hop_size:
                break
            index += 1

        # mfccの1次元はいらないから消す
        mfccs = np.delete(mfccs, 0, axis=1)

        energies = np.mean(energies, axis=1)

        # 対数パワー項を末尾に追加
        mfccs = np.hstack((mfccs, energies.reshape(energies.shape[0], 1)))

        deltas = np.diff(mfccs, axis=0)
        # 先頭行に1行追加し0でパディング
        deltas = np.pad(deltas, [(1, 0), (0, 0)], "constant")

        ddeltas = np.diff(deltas, axis=0)
        ddeltas = np.pad(ddeltas, [(1, 0), (0, 0)], "constant")

        mfccs = mfccs.transpose()
        deltas = deltas.transpose()
        ddeltas = ddeltas.transpose()

        # 正規化なし
        # mfccs = sp.stats.zscore(mfccs, axis=1) # axis=0 or axis=1
        # deltas = sp.stats.zscore(deltas, axis=1)
        # ddeltas = sp.stats.zscore(ddeltas, axis=1)
        # mfccs = preprocessing.minmax_scale(mfccs, axis=1) # 0 ~ 1
        # deltas = preprocessing.minmax_scale(deltas, axis=1)
        # ddeltas = preprocessing.minmax_scale(ddeltas, axis=1)

        mfccs = np.nan_to_num(mfccs)
        deltas = np.nan_to_num(deltas)
        ddeltas = np.nan_to_num(ddeltas)

        all_features = np.concatenate([mfccs, deltas, ddeltas])
        print("Get MFCC in " + file_path + " ...")

    if is_pitch:

        from aubio import pitch

        src = source(file_path, samplerate, hop_size)
        samplerate = src.samplerate

        tolerance = 0.8
        pitch_o = pitch("default", win_s, hop_size, samplerate)
        pitch_o.set_unit("Hz")
        pitch_o.set_silence(-40)
        pitch_o.set_tolerance(tolerance)
        pitches = []

        while True:

            samples, read = src()
            pitch = pitch_o(samples)[0]
            pitches += [float(pitch)]

            if read < hop_size:
                break

        pitches = np.insert(pitches, 0, 0)

        deltas = np.diff(pitches, axis=0)
        deltas = np.insert(deltas, 0, 0)

        pitches = sp.zscore(pitches)
        deltas = sp.zscore(deltas)

        pitches = np.nan_to_num(pitches)
        deltas = np.nan_to_num(deltas)

        all_features = np.concatenate([all_features, [pitches], [deltas]])
        print("Get Pitch in " + file_path + " ...")

    if is_volume:

        src = source(file_path, samplerate, hop_size)
        volumes = []

        while True:

            samples, read = src()

            # Compute the energy (volume) of the
            # current frame.
            volume = np.sum(samples ** 2) / len(samples)
            # Format the volume output so that at most
            # it has six decimal numbers.
            volume = "{:.6f}".format(volume)
            volumes += [float(volume)]

            if read < hop_size:
                break

        volumes = np.nan_to_num(volumes)
        volumes = np.insert(volumes, 0, 0)

        deltas = np.diff(volumes, axis=0)
        deltas = np.insert(deltas, 0, 0)

        volumes = sp.zscore(volumes)
        deltas = sp.zscore(deltas)

        volumes = np.nan_to_num(volumes)
        deltas = np.nan_to_num(deltas)

        all_features = np.concatenate([all_features, [volumes], [deltas]])
        print("Get Volume in " + file_path + " ...")

    if is_tempo:

        src = source(file_path, samplerate, hop_size)
        samplerate = src.samplerate

        tempo_o = tempo("specdiff", win_s, hop_size, samplerate)
        beats = []

        while True:

            samples, read = src()

            is_beat = tempo_o(samples)
            beats.append(float(is_beat))

            if read < hop_size:
                break

        beats = np.nan_to_num(beats)
        beats = np.insert(beats, 0, 0)
        beats = sp.zscore(beats)
        beats = np.nan_to_num(beats)

        all_features = np.concatenate([all_features, [beats]])
        print("Get Tempo in " + file_path + " ...")

    return all_features


def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]


def section_average(input, division_number):

    if len(input) < division_number:

        print("０埋めで音の長さを規定値に伸長...")

        additions = [0.0] * (division_number - len(input))
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


def flatten_with_any_depth(nested_list):

    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)
        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


def get_edited_feature(file_path):

    features = get_feature(file_path)

    section_averages = []
    for feature in features:
        section_averages.append(section_average(feature, time_series_division_number))

    section_averages = flatten_with_any_depth(section_averages)

    return section_averages


if __name__ == "__main__":

    print("\nMake training & test data from raw wavfile.\n")
    wav_count = 1
    all_wav_count = 0

    for emotion_label in emotion_labels:

        wav_path = raw_wav_dir_path + "/" + emotion_label
        wav_file_paths = glob.glob(wav_path + "/*.wav")
        all_wav_count += len(wav_file_paths)

    X = []
    y = []

    for index, emotion_label in enumerate(emotion_labels):

        wav_path = raw_wav_dir_path + "/" + emotion_label
        wav_file_paths = glob.glob(wav_path + "/*.wav")

        for wav_file_path in wav_file_paths:
            print("{}/{} - {}".format(wav_count, all_wav_count, emotion_label))

            y.append(index)
            features = get_edited_feature(wav_file_path)
            X.append(features)

            print(emotion_label + " => " + os.path.basename(wav_file_path) + " section average was completed...\n")
            wav_count += 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    tests = []
    for X, y in zip(X_test, y_test):
        X.insert(0, y)
        tests.append(X)

    print("\nNow test data write to test.csv.")
    write_csv(test_file_path, tests)

    if os.path.isdir(augment_wav_dir_path) and len(os.listdir(augment_wav_dir_path)) != 0:

        print("\nMake train data from augment wavfile.\n")
        wav_count = 1
        all_wav_count = 0

        for emotion_label in emotion_labels:

            wav_path = augment_wav_dir_path + "/" + emotion_label
            wav_file_paths = glob.glob(wav_path + "/*.wav")
            all_wav_count += len(wav_file_paths)

        X = []
        y = []

        for index, emotion in enumerate(emotion_labels):

            wav_path = augment_wav_dir_path + "/" + emotion
            wav_file_paths = glob.glob(wav_path + "/*.wav")

            for wav_file_path in wav_file_paths:
                print("{}/{} - {}".format(wav_count, all_wav_count, emotion))

                y.append(index)
                features = get_feature(wav_file_path)

                section_averages = []
                for feature in features:
                    section_averages.append(section_average(feature, time_series_division_number))

                section_averages = flatten_with_any_depth(section_averages)
                X.append(section_averages)

                print(emotion + " => " + os.path.basename(wav_file_path) + " section average was completed...\n")
                wav_count += 1

        X_train += X
        y_train += y

    trains = []
    for X, y in zip(X_train, y_train):
        X.insert(0, y)
        trains.append(X)

    print("\nNow training data write to train.csv.")
    write_csv(training_file_path, trains)

    print("\nAll process completed...")
