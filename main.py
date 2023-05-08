import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
import re

input_path = 'dataset/spam_ham_dataset.csv'

dataset = pd.read_csv(input_path)
X = dataset['text']
y = dataset['label_num']

words_df = [preprocessing(text) for text in X]

print(words_df[0])

def preprocessing(text):
    list_of_words = re.split(r"\s+", text)
    clear_words = []
    for word in list_of_words:
        r = re.compile(r"[^a-zA-Z0-9]+")
        word = r.sub("", word)
        clear_words.append(word)
    return clear_words