#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
import shutil
import librosa
import soundfile as sf
import csv


# path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
raw_wav_dir_path = data_dir_path + "/wavs/raws"
output_wav_dir_path = data_dir_path + "/wavs/augment"

# rate
time_stretch_rates = [1.25, 1.5, 1.75, 2.0]
pitch_shift_rates = [-4, -2, 2, 4]

# flag
is_add_white_noise = True
is_position_shift = False
is_time_stretch = False
is_pitch_shift = True


# load a wave data
def load_wave_data(file_path):
    x, fs = librosa.load(file_path, sr=16000)
    return x, fs


# save a wave data
def save_wave_data(x, fs, file_path):
    librosa.output.write_wav(file_path, x, fs)


# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate * np.random.randn(len(x))


# data augmentation: shift sound position in timeframe
def position_shift(x, rate=2):
    return np.roll(x, int(len(x) // rate))


# data augmentation: time stretch sound
def time_stretch(x, rate=1.1):

    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)

    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


def pitch_shift(y, sr, rate=6):
    x = librosa.effects.pitch_shift(y, sr=sr, n_steps=rate)
    return x


def convert_1d_to_2d(l, cols):
    return [l[i:i + cols] for i in range(0, len(l), cols)]


if __name__ == "__main__":

    if os.path.isdir(output_wav_dir_path):
        shutil.rmtree(output_wav_dir_path)

    os.makedirs(output_wav_dir_path)
    print("Training Data Augment Start...")

    words = os.listdir(raw_wav_dir_path)

    with open(data_dir_path + "/emotions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(convert_1d_to_2d(words, 1))

    wav_count = 1
    all_wav_count = 0

    for word in words:

        wav_path = raw_wav_dir_path + "/" + word
        wav_file_paths = glob.glob(wav_path + "/*.wav")
        all_wav_count += len(wav_file_paths)

    for word in words:

        wav_path = raw_wav_dir_path + "/" + word
        wav_file_paths = glob.glob("{}/*.wav".format(wav_path))

        _output_wav_dir_path = output_wav_dir_path + "/" + word

        for wav_file_path in wav_file_paths:

            if not os.path.isdir(_output_wav_dir_path):
                os.mkdir(_output_wav_dir_path)

            x, fs = load_wave_data(wav_file_path)
            filename = os.path.basename(wav_file_path)

            if is_add_white_noise:

                x_wn = add_white_noise(x)
                print("{}/{} - {}: Add white noise {}.".format(wav_count, all_wav_count, word, wav_file_path))

                save_file_path = _output_wav_dir_path + "/augment_white_noise_" + filename
                sf.write(save_file_path, x_wn, fs)

            if is_position_shift:

                x_ss = position_shift(x)
                print("{}/{} - {}: Position shifted {}.".format(wav_count, all_wav_count, word, wav_file_path))

                save_file_path = _output_wav_dir_path + "/augment_position_shift_" + filename
                sf.write(save_file_path, x_ss, fs)

            if is_time_stretch:

                for time_stretch_rate in time_stretch_rates:

                    x_st = time_stretch(x, time_stretch_rate)
                    print("{}/{} - {}: Time stretched {}.".format(wav_count, all_wav_count, word, wav_file_path))

                    save_file_path = _output_wav_dir_path + "/time_stretch_rate_" + str(time_stretch_rate) + "_" + filename
                    sf.write(save_file_path, x_st, fs)

            if is_pitch_shift:

                for pitch_shift_rate in pitch_shift_rates:

                    x_pt = pitch_shift(x, fs, pitch_shift_rate)
                    print("{}/{} - {}: Pitch shifted {}.".format(wav_count, all_wav_count, word, wav_file_path))

                    save_file_path = _output_wav_dir_path + "/pitch_shift_rate_" + str(pitch_shift_rate) + "_" + filename
                    sf.write(save_file_path, x_pt, fs)

            wav_count += 1

    print("\nAll process completed...")
