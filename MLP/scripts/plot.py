#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import sqlite3
import os


window_title = "Graph View of Voice Classification"

# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
database_file_path = data_dir_path + "/evaluation.sqlite3"
graph_dir_path = data_dir_path + "/graphs"


def learning_curve(datas, xlabel, ylabel, title, text):

    plt.clf()
    plt.style.use("ggplot")

    xlange = range(len(datas))
    plt.plot(xlange, datas, linewidth=3, linestyle="solid") # dashdot

    # ラベルの追加
    plt.title(title, fontsize=20) # タイトル
    plt.ylabel(ylabel, fontsize=20) # Y 軸
    plt.xlabel(xlabel, fontsize=20) # X 軸

    plt.text(len(datas) * 0.1, max(datas) * 0.8, text, fontsize=14, color='green')

    # 凡例
    # pylab.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()

    # グリッド有効
    # pylab.grid(True)

    # ウィンドウタイトル
    plt.gcf().canvas.set_window_title(window_title)

    # svgに保存
    plt.savefig(graph_dir_path + "/" + title.lower().replace(" ", "_") + ".svg", format="svg")

    # 描画
    plt.show()

    # プロットを表示するためにポップアップしたウィンドウをクローズ
    # このプロジェクトのみの対処方法
    plt.close()


def get_from_database(database_file_path, sql):

    db = sqlite3.connect(database_file_path)
    cur = db.cursor()
    cur.execute(sql)

    outputs = []
    for row in cur:
        outputs.append(row[0])

    cur.close()
    db.close()

    return outputs


if __name__ == "__main__":

    if not os.path.isdir(graph_dir_path):
        os.makedirs(graph_dir_path)

    # プロットするカラムを設定
    results = ["training_accuracy", "validation_accuracy", "training_loss", "validation_loss"]

    # グラフのタイトル
    titles = ["Train accuracy", "Valid accuracy", "Train loss", "Valid loss"]

    # yラベル
    ylabels = ["Accuracy", "Accuracy", "Loss", "Loss"]

    max_min_labels = ["Max train accuracy value", "Max valid accuracy value", "Min train loss", "Min train loss"]

    for index, result in enumerate(results):

        sql = "SELECT %s FROM learning;" % result
        datas = get_from_database(database_file_path, sql)

        # NoneをListから削除
        # datas = [item for item in datas if item is not None]

        if index == 0 or index == 1:
            text = "{}: {}, index: {}".format(max_min_labels[index], round(max(datas), 2), datas.index(max(datas)))

        if index == 2 or index == 3:
            text = "{}: {}, index: {}".format(max_min_labels[index], round(min(datas), 2), datas.index(min(datas)))

        learning_curve(datas, "Epoch", ylabels[index], titles[index], text)

    print("\nAll process completed...")
