#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
import shutil
import librosa
import soundfile as sf


# path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
raw_wav_dir_path = base_absolute_path + "wavs/raws"
output_wav_dir_path = base_absolute_path + "wavs/augment"
broken_wav_dir_path = base_absolute_path + "wavs/broken"

# rate
add_white_noise_rates = [0.002, 0.003]
position_shift_rates = [2, 3]
time_stretch_rates = [0.8, 1.2] # [1.25, 1.5, 1.75, 2.0]
pitch_shift_rates = [-3, 3] # [-4, -2, 2, 4]

# flag
is_add_white_noise = True
is_position_shift = True
is_time_stretch = True
is_pitch_shift = True
is_file_convert_only = False


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
    x = librosa.effects.time_stretch(y=x, rate=rate)

    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


def pitch_shift(y, sr, rate=6):
    x = librosa.effects.pitch_shift(y, sr=sr, n_steps=rate)
    return x


if __name__ == "__main__":

    if os.path.isdir(output_wav_dir_path):
        shutil.rmtree(output_wav_dir_path)

    os.makedirs(output_wav_dir_path)
    print("Training Data Augment Start...")

    words = os.listdir(raw_wav_dir_path)

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

            filename = os.path.basename(wav_file_path)

            try:
                x, fs = load_wave_data(wav_file_path)
            except Exception as e:

                print("\n{}".format(e))
                print("{}: {} is broken!!!\n".format(word, filename))

                if not os.path.isdir(broken_wav_dir_path):
                    os.makedirs(broken_wav_dir_path)

                shutil.move(wav_file_path, broken_wav_dir_path + "/" + filename)
                wav_count += 1
                continue

            if is_add_white_noise:

                for add_white_noise_rate in add_white_noise_rates:

                    x_wn = add_white_noise(x, add_white_noise_rate)
                    print("{}/{} - {}: Add white noise {} rate {}.".format(wav_count, all_wav_count, word, wav_file_path, add_white_noise_rate))

                    save_file_path = _output_wav_dir_path + "/add_white_noise_rate_" + str(add_white_noise_rate) + "_" + filename
                    sf.write(save_file_path, x_wn, fs)

            if is_position_shift:

                for position_shift_rate in position_shift_rates:

                    x_ss = position_shift(x, position_shift_rate)
                    print("{}/{} - {}: Position shifted {} rate {}.".format(wav_count, all_wav_count, word, wav_file_path, position_shift_rate))

                    save_file_path = _output_wav_dir_path + "/position_shift_rate_" + str(position_shift_rate) + "_" +filename
                    sf.write(save_file_path, x_ss, fs)

            if is_time_stretch:

                for time_stretch_rate in time_stretch_rates:

                    x_st = time_stretch(x, time_stretch_rate)
                    print("{}/{} - {}: Time stretched {} rate {}.".format(wav_count, all_wav_count, word, wav_file_path, time_stretch_rate))

                    save_file_path = _output_wav_dir_path + "/time_stretch_rate_" + str(time_stretch_rate) + "_" + filename
                    sf.write(save_file_path, x_st, fs)

            if is_pitch_shift:

                for pitch_shift_rate in pitch_shift_rates:

                    x_pt = pitch_shift(x, fs, pitch_shift_rate)
                    print("{}/{} - {}: Pitch shifted {} rate {}.".format(wav_count, all_wav_count, word, wav_file_path, pitch_shift_rate))

                    save_file_path = _output_wav_dir_path + "/pitch_shift_rate_" + str(pitch_shift_rate) + "_" + filename
                    sf.write(save_file_path, x_pt, fs)

            if is_file_convert_only:

                save_file_path = _output_wav_dir_path + "/" + filename
                sf.write(save_file_path, x, fs)
                print("{}/{} - {}: Wavfile converted {}.".format(wav_count, all_wav_count, word, wav_file_path))

            wav_count += 1

    print("\nAll process completed...")
