# import libraries
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from scipy.special import softmax
from typing import Tuple, Any, List
import numpy as np
import pandas as pd
import torch
import spacy
import os
import re


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(sentence: str):
    sentence = preprocess(sentence)
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return ranking[0]

def process_sentences(df):
    # Crear la columna 'sentence' uniendo los tokens
    df["sentence"] = df["tokens"].apply(lambda x: " ".join(x))

    # Aplicar análisis de sentimiento
    df["SA"] = df["sentence"].apply(analyze_sentiment)
    
    return df

def download_roberta_data():
    ds = load_dataset("tner/ontonotes5")

    df_train = ds["train"].to_pandas()
    df_val = ds["validation"].to_pandas()
    df_test = ds["test"].to_pandas()

    df_train = process_sentences(df_train)
    df_val = process_sentences(df_val)
    df_test = process_sentences(df_test)

    if not os.path.exists("data"):
        os.makedirs("data")

    df_train.to_csv(f"data/train.csv", index=False)
    df_val.to_csv(f"data/val.csv", index=False)
    df_test.to_csv(f"data/test.csv", index=False)

def download_data():
    pass