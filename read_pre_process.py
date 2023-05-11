import pandas as pd
import re
import json
import os

# 0 - ham
# 1 - spam

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

dataset_path = 'dataset/'
input_path = 'input/'

for set in os.listdir(dataset_path):
    print(set)

    dataset = pd.read_csv(f"{dataset_path}{set}")   
    texts = dataset['text']
    y = dataset['label_num']

    tokens_contents = [preprocessing(text) for text in texts]

    dictionary = {
        "label": y.to_list(),
        "content": tokens_contents
    }

    json_object = json.dumps(dictionary, indent=3)
    
    with open(f"{input_path}{set.split('.')[0]}.json", "w") as outfile:
        outfile.write(json_object)