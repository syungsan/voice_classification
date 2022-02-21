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
import scipy.stats as sp
import os
import csv
import shutil
import sqlite3
# import tensorflow as tf
import feature as feat

"""
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
"""

# GPUの無効化
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
training_file_path = data_dir_path + "/train.csv"
feat_time_csv_path = data_dir_path + "/feat_time.csv"
log_dir_path = data_dir_path + "/logs"
database_path = data_dir_path + "/evaluation.sqlite3"

# model_names = ["Simple_RNN", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "CNN_RNN_BiLSTM_Stack"]
model_names = ["CNN_RNN_BiLSTM_Stack"]

nb_classes = len(feat.emotion_labels)

# 1, 32, 64, 128, 256, 512
batch_size = 256

# 学習回数
epochs = 1

# RNN層のユニット数 default 20
hidden_neuron_number = 64

# クロスバリデーションの分割数
# { reference equation: k = log(n) / log(2) }
folds_number = 2

is_smote = True


def load_data(file_name):
    # load your data using this function

    # CSVの制限を外す
    # csv.field_size_limit(sys.maxsize)

    data = []
    target = []

    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=",")

        for columns in reader:
            target.append(columns[0])

            # こいつが決め手か？！
            data.append(columns[1:])

    data = np.array(data, dtype=np.float32)

    # なぜか進数のエラーを返すので処理
    target10b = []
    for tar in target:
        target10b.append(int(float(tar)))

    target = np.array(target10b, dtype=np.int32)

    return data, target


def create_model(model_name):
    # create your model using this function

    model = Sequential()

    if model_name == "Simple_RNN":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(SimpleRNN(hidden_neuron_number, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

    if model_name == "LSTM":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(LSTM(hidden_neuron_number, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

    if model_name == "GRU":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(GRU(hidden_neuron_number, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

    if model_name == "Bidirectional_LSTM":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(Bidirectional(LSTM(hidden_neuron_number, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

    if model_name == "Bidirectional_GRU":

        model.add(Reshape((feat.time_series_division_number, feat.feature_max_length),
                          input_shape=(feat.time_series_division_number * feat.feature_max_length,)))

        model.add(Bidirectional(GRU(hidden_neuron_number, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

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

        model.add(Bidirectional(LSTM(hidden_neuron_number, return_sequences=True)))
        model.add(Bidirectional(LSTM(hidden_neuron_number, return_sequences=True, dropout=0.1, recurrent_dropout=0.5)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.45))

        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

    return model


if __name__ == "__main__":

    # オリジナル特徴量の流入
    X_train, y_train = load_data(file_name=training_file_path)
    X_train = np.nan_to_num(X_train)

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

    if os.path.isdir(log_dir_path):
        shutil.rmtree(log_dir_path)

    os.makedirs(log_dir_path)

    _y_train = y_train

    # convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(y_train)

    # Cross Validation Evaluation Method

    # define X-fold cross validation
    kf = StratifiedKFold(n_splits=folds_number, shuffle=True)
    mdl = 0

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []

    for model_name in model_names:

        cvscores = []
        fld = 0

        _train_accs = [[0 for i in range(folds_number)] for j in range(epochs)]
        _valid_accs = [[0 for i in range(folds_number)] for j in range(epochs)]
        _train_losses = [[0 for i in range(folds_number)] for j in range(epochs)]
        _valid_losses = [[0 for i in range(folds_number)] for j in range(epochs)]

        # cross validation
        for train, test in kf.split(X_train, _y_train):

            model_proccess = mdl / len(model_names) * 100
            fold_proccess = 100 / len(model_names) * fld / folds_number

            fold_proccess = fld / folds_number * 100
            print("\nProccess was {}% completed.".format(fold_proccess))
            print("Model Name is {}.".format(model_name))
            print("Running Fold {}/{}".format(fld + 1, folds_number))

            model = None
            model = create_model(model_name=model_name)

            if not os.path.isdir(log_dir_path):
                os.makedirs(log_dir_path)

            fpath = log_dir_path + "/" + model_name + "/process_" + model_name + "_{0:02d}".format(fld + 1) + \
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
            model.save(log_dir_path + "/" + model_name + "/final_" + model_name + "_{0:02d}".format(fld + 1) +
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
