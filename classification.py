import json
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords

# 0 - ham
# 1 - spam

clfs = {
    "RFC": RandomForestClassifier(),
    "ABC": AdaBoostClassifier(),
    "GBC": GradientBoostingClassifier()
}
 
with open('result.json', 'r') as openfile:
    json_object = json.load(openfile)

labels = json_object['label']
contents = json_object['content']
contents_joined = [" ".join(s) for s in contents]

X_train, X_test, y_train, y_test = train_test_split(contents_joined, labels, test_size=0.33, random_state=10)

# TF - IDF
tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(X_train)
print(X_train_tf[0].shape)
X_train_tf = tf_idf.transform(X_train)

X_test_tf = tf_idf.transform(X_test)
print(X_test_tf[0].shape)



for clf in clfs:
    clf = clf.fit(X_train_tf, y_train)
    y_pred = clf.predict(X_test_tf)
    print(metrics.classification_report(y_test, y_pred, target_names=['Positive', 'Negative']))
