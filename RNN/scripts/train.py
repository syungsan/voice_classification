#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM, GRU
from keras.layers import Bidirectional, MaxPooling1D, Conv1D, Reshape, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.stats as sp
import os
import csv
import shutil
import sqlite3
import tensorflow as tf
import feature as feat


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
training_file_path = data_dir_path + "/train.csv"
# max_mean_csv_path = data_dir_path + "/max_mean.csv"
log_dir_path = data_dir_path + "/logs"
database_path = data_dir_path + "/evaluation.sqlite3"

# model_names = ["DDMLP", "Simple_RNN", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "CNN_RNN_BiLSTM_Stack"]
model_names = ["DDMLP", "LSTM"]

nb_classes = len(feat.emotion_labels)

# 1, 32, 64, 128, 256, 512
batch_size = 128

# 学習回数
epochs = 100

# RNN層のユニット数 default 20
rnn_hidden_unit_number = 64

# {0: closs_validation, 1: hold_out}}
validation_type = 1

if validation_type == 0:
    # クロスバリデーションの分割数
    # { reference equation: k = log(n) / log(2) }
    folds_number = 10

elif validation_type == 1:
    test_size_ratio = 0.1

is_gpu = True

if is_gpu:
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    # GPUの無効化
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

is_smote = True


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


def create_model(model_name):
    # create your model using this function

    model = Sequential()

    if model_name == "DDMLP":

        # Here's a Deep Dumb MLP (DDMLP)
        model.add(Dense(128, input_dim=feat.time_series_division_number * feat.feature_max_length))
        model.add(Activation("relu"))
        model.add(Dropout(0.15))
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dropout(0.15))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        # we'll use categorical xent for the loss, and RMSprop as the optimizer
        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    if model_name == "Simple_RNN":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(SimpleRNN(rnn_hidden_unit_number, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    if model_name == "LSTM":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(LSTM(rnn_hidden_unit_number, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        # model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    if model_name == "GRU":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(GRU(rnn_hidden_unit_number, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    if model_name == "Bidirectional_LSTM":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(Bidirectional(LSTM(rnn_hidden_unit_number, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    if model_name == "Bidirectional_GRU":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(Bidirectional(GRU(rnn_hidden_unit_number, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    if model_name == "CNN_RNN_BiLSTM_Stack":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.3))

        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.35))

        model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.4))

        model.add(Bidirectional(LSTM(rnn_hidden_unit_number, return_sequences=True)))
        model.add(Bidirectional(LSTM(rnn_hidden_unit_number, return_sequences=True, dropout=0.1, recurrent_dropout=0.5)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.45))

        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model


if __name__ == "__main__":

    # Read data
    train = pd.read_csv(training_file_path)
    y_train = train.iloc[:, 0].values.astype("int32")
    X_train = (train.iloc[:, 1:].values).astype("float32")

    if is_smote:
        smote = SMOTE(random_state=42)

        X_train, y_train = smote.fit_resample(X_train, y_train)
        for i in range(nb_classes):
            print("Number of data resampled => {}: {}".format(i, np.sum(y_train == i)))

    # 時系列に変換
    X_train = np.reshape(X_train, (X_train.shape[0], feat.time_series_division_number, feat.feature_max_length), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X_train = np.reshape(X_train, (X_train.shape[0], (feat.time_series_division_number * feat.feature_max_length)))

    # Z-Score関数による正規化
    X_train = sp.zscore(X_train, axis=1)

    """
    # pre-processing: divide by max and substract mean
    scale = np.max(X_train)
    X_train /= scale

    mean = np.std(X_train)
    X_train -= mean

    write_csv(max_mean_csv_path, [[scale, mean]])
    """

    if os.path.isdir(log_dir_path):
        shutil.rmtree(log_dir_path)

    os.makedirs(log_dir_path)

    if validation_type == 0:
        # define X-fold cross validation
        kf = StratifiedKFold(n_splits=folds_number, shuffle=True)

    if not os.path.isdir(log_dir_path):
        os.makedirs(log_dir_path)

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []

    cvscores = []
    mdl = 0

    _y_train = y_train

    # convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(y_train)

    for model_name in model_names:

        print("\nModel Name is {}.".format(model_name))
        print("Running model {}/{}".format(mdl + 1, len(model_names)))

        # Cross validation evaluation method
        if validation_type == 0:

            _train_accs = [[0 for i in range(folds_number)] for j in range(epochs)]
            _valid_accs = [[0 for i in range(folds_number)] for j in range(epochs)]
            _train_losses = [[0 for i in range(folds_number)] for j in range(epochs)]
            _valid_losses = [[0 for i in range(folds_number)] for j in range(epochs)]
            fld = 0

            for train, test in kf.split(X_train, _y_train):

                proccess = 100 / len(model_names) * fld / folds_number
                proccess = fld / folds_number * 100
                print("\nAll proccess was {}% completed.".format(proccess))
                print("Running fold {}/{}\n".format(fld + 1, folds_number))

                model = None
                model = create_model(model_name=model_name)

                fpath = log_dir_path + "/" + model_name + "/process_" + model_name + "_fold-{0:02d}".format(fld + 1) + \
                        "_{epoch:02d}-{loss:.2f}-{val_loss:.2f}-{accuracy:.2f}-{val_accuracy:.2f}.h5"

                model_checkpoint = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=False,
                                                   save_weights_only=False, mode='min', period=0)

                hist = model.fit(X_train[train], y_train[train], validation_data=(X_train[test], y_train[test]),
                                 batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[model_checkpoint])

                # Evaluate
                scores = model.evaluate(X_train[test], y_train[test], verbose=1)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                cvscores.append(scores[1] * 100)

                for i in model.layers:
                    if type(i) is Dropout:
                        model.layers.remove(i)

                # Save the model
                model.save(log_dir_path + "/" + model_name + "/final_" + model_name + "_fold-{0:02d}".format(fld + 1) +
                           "_{}-{}%_model.h5".format(round(scores[0], 2), round(scores[1], 2)))

                for epoch in range(epochs):
                    _train_accs[epoch][fld] = hist.history["accuracy"][epoch]
                    _valid_accs[epoch][fld] = hist.history["val_accuracy"][epoch]
                    _train_losses[epoch][fld] = hist.history["loss"][epoch]
                    _valid_losses[epoch][fld] = hist.history["val_loss"][epoch]

                fld += 1

            train_acc_means = []
            valid_acc_means = []
            train_loss_means = []
            valid_loss_means = []

            for epoch in range(epochs):
                train_acc_means.append(np.mean(_train_accs[epoch]))
                valid_acc_means.append(np.mean(_valid_accs[epoch]))
                train_loss_means.append(np.mean(_train_losses[epoch]))
                valid_loss_means.append(np.mean(_valid_losses[epoch]))

            train_accs.append(train_acc_means)
            valid_accs.append(valid_acc_means)
            train_losses.append(train_loss_means)
            valid_losses.append(valid_loss_means)

        # Hold out evaluation method
        elif validation_type == 1:

            proccess = mdl / len(model_names) * 100
            print("\nAll proccess was {}% completed.".format(proccess))

            model = None
            model = create_model(model_name=model_name)

            fpath = log_dir_path + "/" + model_name + "/process_" + model_name + "_hold" + \
                    "_{epoch:02d}-{loss:.2f}-{val_loss:.2f}-{accuracy:.2f}-{val_accuracy:.2f}.h5"

            model_checkpoint = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=False,
                                               save_weights_only=False, mode='min', period=0)

            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size_ratio)

            hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                             batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[model_checkpoint])

            # Evaluate
            scores = model.evaluate(X_test, y_test, verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            for i in model.layers:
                if type(i) is Dropout:
                    model.layers.remove(i)

            # Save the model
            model.save(log_dir_path + "/" + model_name + "/final_" + model_name + "_hold" +
                       "_{}-{}%_model.h5".format(round(scores[0], 2), round(scores[1], 2)))

            train_accs.append(hist.history["accuracy"])
            valid_accs.append(hist.history["val_accuracy"])
            train_losses.append(hist.history["loss"])
            valid_losses.append(hist.history["val_loss"])

        print("\n%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        mdl += 1

    if os.path.exists(database_path):
        os.remove(database_path)

    db = sqlite3.connect(database_path)
    cur = db.cursor()

    sql = "CREATE TABLE IF NOT EXISTS learning (model TEXT, epoch INTEGER, training_accuracy REAL, validation_accuracy REAL, training_loss REAL, validation_loss REAL);"

    cur.execute(sql)
    db.commit()

    datas = []
    for i in range(len(model_names)):
        for j in range(len(hist.epoch)):

            # なぜか3番目のtrain accuracyがバイナリーになるのでfloatにキャストし直し（sqliteのバージョンによるバグ？）
            datas.append((model_names[i], j + 1, float(train_accs[i][j]), valid_accs[i][j], train_losses[i][j], valid_losses[i][j]))

    sql = "INSERT INTO learning (model, epoch, training_accuracy, validation_accuracy, training_loss, validation_loss) VALUES (?, ?, ?, ?, ?, ?);"

    cur.executemany(sql, datas)
    db.commit()

    cur.close()
    db.close()

    print("\nAll proccess completed...")
