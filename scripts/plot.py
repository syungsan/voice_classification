#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import sqlite3
import os
import train as tr


window_title = "Graph View of Voice Classification"

# Path
base_absolute_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
data_dir_path = base_absolute_path + "data"
database_file_path = data_dir_path + "/evaluation.sqlite3"
graph_dir_path = data_dir_path + "/graphs"


def learning_curve(datas, line_names, title, xlabel, ylabel, text, text_y_pos):

    plt.clf()
    plt.style.use("ggplot")

    for i in range(0, len(datas)):

        x = range(len(datas[i]))
        plt.plot(x, datas[i], label=line_names[i], linewidth=3, linestyle="solid") # dashdot

    # pylab.ylim(0.0, 1.2)

    # ラベルの追加
    plt.title(title, fontsize=20) # タイトル
    plt.ylabel(ylabel, fontsize=20) # Y 軸
    plt.xlabel(xlabel, fontsize=20) # X 軸

    plt.text(0.1, text_y_pos, text)

    # 凡例
    # pylab.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()

    # グリッド有効
    # pylab.grid(True)

    # ウィンドウタイトル
    plt.gcf().canvas.set_window_title(window_title)

    # svgに保存
    plt.savefig(graph_dir_path + "/" + title.lower().replace(" ", "_") + ".svg", format="svg")

    # 描画
    plt.show()


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

    datas = []
    for i in range(len(results)):

        line = []
        for j in range(len(tr.model_names)):

            sql = "SELECT %s FROM learning WHERE model = '%s';" % (results[i], tr.model_names[j])
            data = get_from_database(database_file_path=database_file_path, sql=sql)

            # NoneをListから削除
            cut_data = [item for item in data if item is not None]

            line.append(cut_data)

        datas.append(line)

    for k in range(len(results)):

        rate = 0.0
        index = 0
        model = ""
        text = ""

        if k == 0 or k == 1:

            l = 0
            max_datas = []
            max_factors = []

            for data in datas[k]:

                # maxのindexを全返し
                # [m for m, x in enumerate(data) if x == max(data)]

                max_datas.append(max(data))
                max_factors.append([max(data), [i for i, x in enumerate(data) if x == max(data)][0], tr.model_names[l]])

                if k == 0:
                    print("Max train accuracy => model: %s; value: %f; index: %d" % (tr.model_names[l], max(data), data.index(max(data))))
                if k == 1:
                    print("Max valid accuracy => model: %s; value: %f; index: %d" % (tr.model_names[l], max(data), data.index(max(data))))

                l += 1

            rate = max_factors[[i for i, x in enumerate(max_datas) if x == max(max_datas)][0]][0]
            index = max_factors[[i for i, x in enumerate(max_datas) if x == max(max_datas)][0]][1]
            model = max_factors[[i for i, x in enumerate(max_datas) if x == max(max_datas)][0]][2]

            text = "Max " + results[k] + " = " + str(rate * 100.0) + "%, " + model + ", " + str(index) + "epoch"

        if k == 2 or k == 3:

            l = 0
            min_datas = []
            min_factors = []

            for data in datas[k]:

                min_datas.append(min(data))
                min_factors.append([min(data), [i for i, x in enumerate(data) if x == min(data)][0], tr.model_names[l]])

                if k == 2:
                    print("Min train loss => model: %s; value: %f; index: %d" % (tr.model_names[l], min(data), data.index(min(data))))
                if k == 3:
                    print("Min valid loss => model: %s; value: %f; index: %d" % (tr.model_names[l], min(data), data.index(min(data))))

                l += 1

            rate = min_factors[[i for i, x in enumerate(min_datas) if x == min(min_datas)][0]][0]
            index = min_factors[[i for i, x in enumerate(min_datas) if x == min(min_datas)][0]][1]
            model = min_factors[[i for i, x in enumerate(min_datas) if x == min(min_datas)][0]][2]

            text = "Min " + results[k] + " = " + str(rate) + "%, " + model + ", " + str(index) + "epoch"

        learning_curve(datas[k], tr.model_names, titles[k], "Epoch", ylabels[k], text, rate)

    print("\nAll process completed...")
