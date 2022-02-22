#!/usr/bin/env python
# coding: utf-8

# import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import os
import csv
import glob
import pandas as pd
import scipy.stats as sp
import shutil
import tensorflow as tf
# import feature as feat
import train as tr


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
test_file_path = data_dir_path + "/test.csv"
# max_mean_csv_path = data_dir_path + "/max_mean.csv"
best_models_save_dir_path = data_dir_path + "/best_models"
best_models_score_csv_path = data_dir_path + "/best_models/score.csv"
log_dir_path = data_dir_path + "/logs"

is_gpu = True

if is_gpu:
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    # GPUの無効化
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


if __name__ == "__main__":

    # Read data
    test = pd.read_csv(test_file_path)
    y_test = test.iloc[:, 0].values.astype("int32")
    X_test = (test.iloc[:, 1:].values).astype("float32")

    # 時系列に変換
    # X_test = np.reshape(X_test, (X_test.shape[0], feat.time_series_division_number, feat.feature_max_length), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    # X_test = np.reshape(X_test, (X_test.shape[0], (feat.time_series_division_number * feat.feature_max_length)))

    # Z-Score関数による正規化
    X_test = sp.zscore(X_test, axis=1)

    y_test = np_utils.to_categorical(y_test)

    if os.path.isdir(best_models_save_dir_path):
        shutil.rmtree(best_models_save_dir_path)

    os.makedirs(best_models_save_dir_path)

    best_model_scores = []
    best_model_scores.append(["accuracy", "loss", "best_model_name"])

    for model_name in tr.model_names:

        evaluations = []
        model_paths = glob.glob(log_dir_path + "/" + model_name + "/*.h5")

        test_length = len(model_paths)
        test_count = 1

        for model_path in model_paths:

            model = load_model(model_path)

            score = model.evaluate(X_test, y_test, verbose=1)
            base_model_name = os.path.basename(model_path)

            print("\nModel Detail Name: {}".format(base_model_name))
            print("Test accuracy: {}".format(score[1]))
            print("Test loss: {}".format(score[0]))

            evaluations.append([score[1], score[0], base_model_name])
            model = None

            print("Test = {}/{} completed...\n".format(test_count, test_length))
            test_count += 1

        max_accuracy = max(evaluations)[0]
        accuracy_maxs = [i for i in evaluations if i[0] == max_accuracy]

        min_loss = min([r[1] for r in accuracy_maxs])
        loss_mins = [i for i in accuracy_maxs if i[1] == min_loss]

        best_model_scores.append(loss_mins[0])
        best_model_name = loss_mins[0][2]

        shutil.copy(log_dir_path + "/" + model_name + "/" + best_model_name, best_models_save_dir_path + "/" + best_model_name)

    write_csv(best_models_score_csv_path, best_model_scores)
    shutil.rmtree(log_dir_path)
    print("\nAll process completed...")
