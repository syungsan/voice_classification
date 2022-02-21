#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import os
import csv
import shutil
import itertools
import sqlite3
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
raw_wav_dir_path = base_absolute_path + "../wavs/raws"
output_wav_dir_path = base_absolute_path + "../wavs/augment"
data_dir_path = base_absolute_path + "data"
training_file_path = data_dir_path + "/train.csv"
test_file_path = data_dir_path + "/test.csv"
emotions_csv_path = data_dir_path + "/emotions.csv"
max_mean_csv_path = data_dir_path + "/max_mean.csv"
log_dir_path = data_dir_path + "/logs"
database_path = data_dir_path + "/evaluation.sqlite3"

# クロスバリデーションの分割数
# { reference equation: k = log(n) / log(2) }
folds_number = 10

is_smote = True

batch_size = 32
epochs = 500


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


def create_model(input_dim, nb_classes):

    # Here's a Deep Dumb MLP (DDMLP)
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation("relu"))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))

    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model


if __name__ == "__main__":

    emotions = []
    with open(emotions_csv_path) as f:

        reader = csv.reader(f, delimiter=",")
        for row in reader:
            emotions.append(row)

    emotions = list(itertools.chain.from_iterable(emotions))

    # Read data
    train = pd.read_csv(training_file_path)
    y_train = train.iloc[:, 0].values.astype("int32")
    X_train = (train.iloc[:, 1:].values).astype("float32")

    input_dim = X_train.shape[1]
    nb_classes = np.max(y_train) + 1

    if is_smote:
        smote = SMOTE(random_state=42)

        X_train, y_train = smote.fit_resample(X_train, y_train)
        for i in range(nb_classes):
            print("Number of data resampled => {}: {}".format(i, np.sum(y_train == i)))

    # pre-processing: divide by max and substract mean
    scale = np.max(X_train)
    X_train /= scale

    mean = np.std(X_train)
    X_train -= mean

    write_csv(max_mean_csv_path, [[scale, mean]])

    if os.path.isdir(log_dir_path):
        shutil.rmtree(log_dir_path)

    os.makedirs(log_dir_path)

    # define X-fold cross validation
    kf = StratifiedKFold(n_splits=folds_number, shuffle=True)

    cvscores = []
    fld = 0

    _train_acc = [[0 for i in range(folds_number)] for j in range(epochs)]
    _valid_acc = [[0 for i in range(folds_number)] for j in range(epochs)]
    _train_loss = [[0 for i in range(folds_number)] for j in range(epochs)]
    _valid_loss = [[0 for i in range(folds_number)] for j in range(epochs)]

    _y_train = y_train

    # convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(y_train)

    # cross validation
    for train, test in kf.split(X_train, _y_train):

        model = None
        model = create_model(input_dim=input_dim, nb_classes=nb_classes)

        fold_proccess = fld / folds_number * 100
        print("\nProccess was {}% completed.".format(fold_proccess))
        print("Running Fold {}/{}".format(fld + 1, folds_number))

        if not os.path.isdir(log_dir_path):
            os.makedirs(log_dir_path)

        fpath = log_dir_path + "/process_model_{0:02d}".format(fld + 1) + \
                "_{epoch:02d}-{loss:.2f}-{val_loss:.2f}-{accuracy:.2f}-{val_accuracy:.2f}.h5"

        model_checkpoint = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=False,
                                           save_weights_only=False, mode='min', period=0)

        hist = model.fit(X_train[train], y_train[train], validation_data=(X_train[test], y_train[test]),
                         batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[model_checkpoint])

        # Evaluate
        scores = model.evaluate(X_train[test], y_train[test], verbose=1)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

        for i in model.layers:
            if type(i) is Dropout:
                model.layers.remove(i)

        # Save the model
        model.save(log_dir_path + "/final_model_{0:02d}".format(fld + 1) +
                   "_{}-{}%_model.h5".format(round(scores[0], 2), round(scores[1], 2)))

        for epoch in range(epochs):
            _train_acc[epoch][fld] = hist.history["accuracy"][epoch]
            _valid_acc[epoch][fld] = hist.history["val_accuracy"][epoch]
            _train_loss[epoch][fld] = hist.history["loss"][epoch]
            _valid_loss[epoch][fld] = hist.history["val_loss"][epoch]

        fld += 1

    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []

    for epoch in range(epochs):
        train_acc.append(np.mean(_train_acc[epoch]))
        valid_acc.append(np.mean(_valid_acc[epoch]))
        train_loss.append(np.mean(_train_loss[epoch]))
        valid_loss.append(np.mean(_valid_loss[epoch]))

    print("\n%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    if os.path.exists(database_path):
        os.remove(database_path)

    db = sqlite3.connect(database_path)
    cur = db.cursor()

    sql = "CREATE TABLE IF NOT EXISTS learning (epoch INTEGER, training_accuracy REAL, validation_accuracy REAL, training_loss REAL, validation_loss REAL);"

    cur.execute(sql)
    db.commit()

    datas = []
    for i in range(len(hist.epoch)):

        # なぜか3番目のtrain accuracyがバイナリーになるのでfloatにキャストし直し（sqliteのバージョンによるバグ？）
        datas.append((i + 1, train_acc[i], valid_acc[i], train_loss[i], valid_loss[i]))

    sql = "INSERT INTO learning (epoch, training_accuracy, validation_accuracy, training_loss, validation_loss) VALUES (?, ?, ?, ?, ?);"

    cur.executemany(sql, datas)
    db.commit()

    cur.close()
    db.close()

    print("\nAll proccess completed...")
