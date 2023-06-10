import json
from math import pi
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
# 0 - ham
# 1 - spam

folders = {
    "input": "input",
    "input_no_preprocessing": "input_no_preprocessing",

}
clf = RandomForestClassifier()

metrics = {
    "recall": recall_score,
    'precision': precision_score,
    'f1': f1_score,
}

n_datasets = 3
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
scores = []
scores = np.zeros((len(folders), n_datasets, n_splits * n_repeats, len(metrics)))

for folder_id, folder in enumerate(folders):
    for dataset_id, input in enumerate(os.listdir(folder)):
        print(input)
        with open(f"{folder}/{input}", 'r') as openfile:
            json_object = json.load(openfile)

        y = json_object['label']
        contents = json_object['content']
        X = [" ".join(s) for s in contents]
        X = np.array(X)
        y = np.array(y)
        for fold_id, (train, test) in enumerate(rskf.split(X, y)):

        # X_train, X_test, y_train, y_test = train_test_split(contents_joined, labels, test_size=0.33, random_state=10)

            tf_idf = TfidfVectorizer(max_features=2000)
            X_train_tf = tf_idf.fit_transform(X[train])
            X_test_tf = tf_idf.transform(X[test])

            clf.fit(X_train_tf, y[train])
            y_pred = clf.predict(X_test_tf)

            for metric_id, metric in enumerate(metrics):
                scores[folder_id, dataset_id, fold_id, metric_id] = metrics[metric](
                    y[test], y_pred) 
            # print(classification_report(y[test], y_pred, target_names=['Positive', 'Negative']))

np.save('results3', scores)

# //////////////////////////////////////////////////////////////////////////

# scores = np.load('results3.npy')
# print("\nScores:\n", scores.shape)


# scores = np.mean(scores, axis=2).T
# scores = np.mean(scores, axis=1)
# metrics=["Recall", 'Precision', 'F1']
# methods=["RFC", 'ABC', 'GBC']


# N = scores.shape[0]

# # kat dla kazdej z osi
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]

# # spider plot
# ax = plt.subplot(111, polar=True)

# # pierwsza os na gorze
# ax.set_theta_offset(pi / 2)
# ax.set_theta_direction(-1)

# # po jednej osi na metryke
# plt.xticks(angles[:-1], metrics)

# # os y
# ax.set_rlabel_position(0)
# plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
# ["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
# color="grey", size=7)
# plt.ylim(0,1)

# print(methods)
# for method_id, method in enumerate(methods):
#     values=scores[:, method_id].tolist()
#     print(values)
#     values += values[:1]
#     print(values)
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

# # Dodajemy legende
# plt.legend(bbox_to_anchor=(1, -0.05), ncol=5)
# # Zapisujemy wykres
# plt.savefig("radar3", dpi=200)


# scores = np.load('results3.npy')
# scores = np.mean(scores, axis=2).T

# for i in range(scores.shape[0]):
#     print("\nMean scores:\n", scores)
#     from scipy.stats import rankdata
#     ranks = []
#     for ms in scores:
#         ranks.append(rankdata(ms).tolist())
#     ranks = np.array(ranks)
#     print("\nRanks:\n", ranks)

#     from scipy.stats import ranksums
#     alfa = .05
#     w_statistic = np.zeros((len(clfs), len(clfs)))
#     p_value = np.zeros((len(clfs), len(clfs)))

#     for i in range(len(clfs)):
#         for j in range(len(clfs)):
#             w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

#     from tabulate import tabulate

#     headers = list(clfs.keys())
#     names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
#     w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
#     w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
#     p_value_table = np.concatenate((names_column, p_value), axis=1)
#     p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
#     print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

#     advantage = np.zeros((len(clfs), len(clfs)))
#     advantage[w_statistic > 0] = 1
#     advantage_table = tabulate(np.concatenate(
#         (names_column, advantage), axis=1), headers)
#     print("\nAdvantage:\n", advantage_table)

#     significance = np.zeros((len(clfs), len(clfs)))
#     significance[p_value <= alfa] = 1
#     significance_table = tabulate(np.concatenate(
#         (names_column, significance), axis=1), headers)
#     print("\nStatistical significance (alpha = 0.05):\n", significance_table)
