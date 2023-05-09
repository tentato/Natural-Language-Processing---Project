import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
import re

def preprocessing(text):
    list_of_words = re.split(r"\s+", text)
    clear_words = []
    for word in list_of_words:
        r = re.compile(r"[^a-zA-Z0-9]+")
        word = r.sub("", word)
        word = word.lower()
        clear_words.append(word)
    clear_words = list(filter(None, clear_words))
    return clear_words

input_path = 'dataset/spam_ham_dataset.csv'

dataset = pd.read_csv(input_path)
texts = dataset['text']
y = dataset['label_num']

subjects, contents = [text.split("\n")[0] for text in texts], [text.replace(text.split("\n")[0], "") for text in texts]

tokens_subjects, tokens_contents = [preprocessing(text) for text in subjects], [preprocessing(text) for text in contents]

print(tokens_subjects[0])
print(tokens_contents[0])

# result_name = "result.json"
# dictionary = {
#     "ilosc_zdan": sentences_count,
#     "lista_oczyszczonych_zdan": clear_sentences,
#     "ilosc_slow": word_count,
#     "lista_oczyszczonych_slow": clear_words
# }

# json_object = json.dumps(dictionary, indent=4)
 
# with open(result_name, "w") as outfile:
#     outfile.write(json_object)