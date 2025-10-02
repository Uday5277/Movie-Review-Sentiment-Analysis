import pandas as pd
import os

def load_imdb(split_path):
    texts = []
    labels = []
    for label in ["pos","neg"]:
        folder = os.path.join(split_path,label)
        for filename in os.listdir(folder):
            filepath = os.path.join(folder,filename)
            with open(filepath,'r',encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label=='pos' else 0)
    return pd.DataFrame({'text':texts,'label':labels})

