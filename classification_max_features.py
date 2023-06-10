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

clf = RandomForestClassifier()
max_features ={
    "100": 100,
    "500": 500,
    "2000": 2000
}
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
scores = np.zeros((len(max_features), n_datasets, n_splits * n_repeats, len(metrics)))

for dataset_id, input in enumerate(os.listdir("input")):
    print(input)
    with open(f"input/{input}", 'r') as openfile:
        json_object = json.load(openfile)

    y = json_object['label']
    contents = json_object['content']
    X = [" ".join(s) for s in contents]
    X = np.array(X)
    y = np.array(y)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):

    # X_train, X_test, y_train, y_test = train_test_split(contents_joined, labels, test_size=0.33, random_state=10)

        for f_id, f in enumerate(max_features):

            tf_idf = TfidfVectorizer(max_features=f)
            X_train_tf = tf_idf.fit_transform(X[train])
            X_test_tf = tf_idf.transform(X[test])
            print(f)
            clf.fit(X_train_tf, y[train])
            y_pred = clf.predict(X_test_tf)


            for metric_id, metric in enumerate(metrics):
                scores[f_id, dataset_id, fold_id, metric_id] = metrics[metric](
                    y[test], y_pred) 
            # print(classification_report(y[test], y_pred, target_names=['Positive', 'Negative']))

np.save('results', scores)

# //////////////////////////////////////////////////////////////////////////

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)


scores = np.mean(scores, axis=2).T
metrics=["Recall", 'Precision', 'F1']
methods=["RFC", 'ABC', 'GBC']


N = scores.shape[0]

# kat dla kazdej z osi
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# spider plot
ax = plt.subplot(111, polar=True)

# pierwsza os na gorze
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# po jednej osi na metryke
plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)


for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

# Dodajemy legende
plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# Zapisujemy wykres
plt.savefig("radar", dpi=200)