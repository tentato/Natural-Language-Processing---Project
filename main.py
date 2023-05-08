import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords

input_path = 'dataset/spam_ham_dataset.csv'

dataset = pd.read_csv(input_path)
X = dataset['text']
y = dataset['label_num']

print(y)