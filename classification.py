import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords

# 0 - ham
# 1 - spam
 
with open('result.json', 'r') as openfile:
    json_object = json.load(openfile)

labels = json_object['label']
subjects = json_object['subject']
contents = json_object['content']

print(labels[0])
print(subjects[0])
print(contents[0])