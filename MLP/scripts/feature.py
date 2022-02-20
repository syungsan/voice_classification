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
# from sklearn import preprocessing


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
raw_wav_dir_path = data_dir_path + "/wavs/raws"
augment_wav_dir_path = data_dir_path + "/wavs/augment"
training_file_path = data_dir_path + "/train.csv"
test_file_path = data_dir_path + "/test.csv"
emotions_csv_path = data_dir_path + "/emotions.csv"

is_mfcc = True
is_pitch = True
is_volume = True
is_tempo = True


def write_csv(path, list):

    try:
        # 書き込み UTF-8
        with open(path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
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

    all_features = np.array([])

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
                samples = np.pad(samples, [0, hop_size - (total_frames - old_frame)], "constant")
            else:
                samples = total_samples[old_frame:cur_frame]

            spec = p(samples)
            mfcc_out = m(spec)
            mfccs = np.vstack((mfccs, mfcc_out))

            if total_frames - old_frame < hop_size:
                break
            index += 1

        mfccs = np.delete(mfccs, 0, axis=1)
        energies = np.mean(energies, axis=1)
        mfccs = np.hstack((mfccs, energies.reshape(energies.shape[0], 1)))

        deltas = np.diff(mfccs, axis=0)
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

        mfccs = np.mean(mfccs, axis=1)
        deltas = np.mean(deltas, axis=1)
        ddeltas = np.mean(ddeltas, axis=1)

        all_features = np.concatenate([all_features, mfccs, deltas, ddeltas])
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

        features = sp.stats.zscore(pitches)
        # features = preprocessing.minmax_scale(pitches)
        features = np.nan_to_num(features)
        feature = np.mean(features)

        all_features = np.concatenate([all_features, [feature]])
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

        features = sp.stats.zscore(volumes)
        # features = preprocessing.minmax_scale(volumes)
        features = np.nan_to_num(features)
        feature = np.mean(features)

        all_features = np.concatenate([all_features, [feature]])
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

        features = sp.stats.zscore(beats)
        # features = preprocessing.minmax_scale(beats)
        features = np.nan_to_num(features)
        feature = np.mean(features)

        all_features = np.concatenate([all_features, [feature]])
        print("Get Tempo in " + file_path + " ...")

    return all_features


def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]


if __name__ == "__main__":

    emotions = os.listdir(raw_wav_dir_path)

    with open(data_dir_path + "/emotions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(convert_1d_to_2d(emotions, 1))

    print("\nMake training & test data from raw wavfile.\n")
    wav_count = 1
    all_wav_count = 0

    for emotion in emotions:

        wav_path = raw_wav_dir_path + "/" + emotion
        wav_file_paths = glob.glob(wav_path + "/*.wav")
        all_wav_count += len(wav_file_paths)

    X = []
    y = []

    for index, emotion in enumerate(emotions):

        wav_path = raw_wav_dir_path + "/" + emotion
        wav_file_paths = glob.glob(wav_path + "/*.wav")

        for wav_file_path in wav_file_paths:
            print("{}/{} - {}".format(wav_count, all_wav_count, emotion))

            y.append(index)
            features = get_feature(wav_file_path)
            X.append(features.tolist())
            wav_count += 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    tests = []
    for X, y in zip(X_test, y_test):
        X.insert(0, y)
        tests.append(X)

    print("\nNow test data write to test.csv.")
    write_csv(test_file_path, tests)

    print("\nMake train data from augment wavfile.\n")
    wav_count = 1
    all_wav_count = 0

    for emotion in emotions:

        wav_path = augment_wav_dir_path + "/" + emotion
        wav_file_paths = glob.glob(wav_path + "/*.wav")
        all_wav_count += len(wav_file_paths)

    X = []
    y = []

    for index, emotion in enumerate(emotions):

        wav_path = augment_wav_dir_path + "/" + emotion
        wav_file_paths = glob.glob(wav_path + "/*.wav")

        for wav_file_path in wav_file_paths:
            print("{}/{} - {}".format(wav_count, all_wav_count, emotion))

            y.append(index)
            features = get_feature(wav_file_path)
            X.append(features.tolist())
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
