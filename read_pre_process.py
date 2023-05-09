import pandas as pd
import re
import json

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
result_name = "result.json"


dataset = pd.read_csv(input_path)
texts = dataset['text']
y = dataset['label_num']

subjects, contents = [text.split("\n")[0] for text in texts], [text.replace(text.split("\n")[0], "") for text in texts]

tokens_subjects, tokens_contents = [preprocessing(text) for text in subjects], [preprocessing(text) for text in contents]

dictionary = {
    "label": y.to_list(),
    "subject": tokens_subjects,
    "content": tokens_contents
}

json_object = json.dumps(dictionary, indent=3)
 
with open(result_name, "w") as outfile:
    outfile.write(json_object)