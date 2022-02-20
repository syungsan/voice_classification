#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
from keras.utils import np_utils
import os
import csv
import glob
import itertools
import pandas as pd
import shutil


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
test_file_path = data_dir_path + "/test.csv"
max_mean_csv_path = data_dir_path + "/max_mean.csv"
best_model_csv_path = data_dir_path + "/best_model.csv"
log_dir_path = data_dir_path + "/logs"


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

    max_means = []
    with open(max_mean_csv_path) as f:

        reader = csv.reader(f, delimiter=",")
        for row in reader:
            max_means.append(row)

    max_means = list(itertools.chain.from_iterable(max_means))

    # Read data
    test = pd.read_csv(test_file_path)
    y_test = test.iloc[:, 0].values.astype("int32")
    X_test = (test.iloc[:, 1:].values).astype("float32")

    X_test /= float(max_means[0])
    X_test -= float(max_means[1])

    y_test = np_utils.to_categorical(y_test)

    evaluations = []
    model_paths = glob.glob(log_dir_path + "/*.h5")

    test_length = len(model_paths)
    test_count = 1

    for model_path in model_paths:

        model = load_model(model_path)
        # model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

        score = model.evaluate(X_test, y_test, verbose=1)
        model_name = os.path.basename(model_path)

        print("\nModel Detail Name: {}".format(model_name))
        print("Test accuracy: {}".format(score[1]))
        print("Test loss: {}".format(score[0]))

        evaluations.append([score[1], score[0], model_name])
        model = None

        print("Test = {}/{} completed...\n".format(test_count, test_length))
        test_count += 1

    max_accuracy = max(evaluations)[0]
    accuracy_maxs = [i for i in evaluations if i[0] == max_accuracy]

    min_loss = min([r[1] for r in accuracy_maxs])
    loss_mins = [i for i in accuracy_maxs if i[1] == min_loss]

    write_csv(best_model_csv_path, [["accuracy", "loss", "best_model_name"], loss_mins[0]])
    best_model_name = loss_mins[0][2]

    shutil.copy(log_dir_path + "/" + best_model_name, data_dir_path + "/" + best_model_name)
    shutil.rmtree(log_dir_path)

    print("\nAll process completed...")
