#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
import shutil
import librosa
import soundfile as sf


# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_WAV_DIR_PATH = DATA_DIR_PATH + "/wavs/train"
OUTPUT_WAV_DIR_PATH = DATA_DIR_PATH + "/wavs/augment"

TIME_STRETCH_RATES = [1.25, 1.5, 1.75, 2.0]
PITCH_SHIFT_RATES = [-4, -2, 2, 4]

IS_ADD_WHITE_NOISE = True
IS_POSITION_SHIFT = True
IS_TIME_STRETCH = True
IS_PITCH_SHIFT = True


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

    x = librosa.effects.pitch_shift(y, sr, n_steps=rate)
    return x


if __name__ == "__main__":

    if os.path.isdir(OUTPUT_WAV_DIR_PATH):
        shutil.rmtree(OUTPUT_WAV_DIR_PATH)

    os.makedirs(OUTPUT_WAV_DIR_PATH)

    print("Training Data Augment Start...")

    words = os.listdir(TRAINING_WAV_DIR_PATH)

    for word in words:

        wav_path = TRAINING_WAV_DIR_PATH + "/" + word

        # パス内の全ての"指定パス+ファイル名"と"指定パス+ディレクトリ名"を要素とするリストを返す
        wav_file_paths = glob.glob("%s/*.wav" % wav_path)

        output_dir_path = OUTPUT_WAV_DIR_PATH + "/" + word

        for wav_file_path in wav_file_paths:

            if not os.path.isdir(output_dir_path):
                os.mkdir(output_dir_path)

            x, fs = load_wave_data(wav_file_path)

            if IS_ADD_WHITE_NOISE:

                x_wn = add_white_noise(x)
                print("ADD WHITE NOISE " + wav_file_path)

                save_file_path = output_dir_path + "/augment_white_noise_" + os.path.basename(wav_file_path)

                # save_wave_data(x_wn, fs, save_file_path)
                sf.write(save_file_path, x_wn, fs)


            if IS_POSITION_SHIFT:

                # 音声の前後を入れ替えるので正解の音声には適用しない
                x_ss = position_shift(x)
                print("POSITION SHIFT " + wav_file_path)

                save_file_path = output_dir_path + "/augment_position_shift_" + os.path.basename(wav_file_path)
                sf.write(save_file_path, x_ss, fs)


            if IS_TIME_STRETCH:

                for time_stretch_rate in TIME_STRETCH_RATES:

                    x_st = time_stretch(x, time_stretch_rate)
                    print("TIME STRETCH " + wav_file_path)

                    save_file_path = output_dir_path + "/time_stretch_rate_" + str(time_stretch_rate) + "_" + os.path.basename(wav_file_path)
                    sf.write(save_file_path, x_st, fs)

            if IS_PITCH_SHIFT:

                for pitch_shift_rate in PITCH_SHIFT_RATES:

                    x_pt = pitch_shift(x, fs, pitch_shift_rate)
                    print("PITCH SHIFT " + wav_file_path)

                    save_file_path = output_dir_path + "/pitch_shift_rate_" + str(pitch_shift_rate) + "_" + os.path.basename(wav_file_path)
                    sf.write(save_file_path, x_pt, fs)

    print("\nAll process completed...")
