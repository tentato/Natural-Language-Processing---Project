import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
 
with open('result.json', 'r') as openfile:
    json_object = json.load(openfile)