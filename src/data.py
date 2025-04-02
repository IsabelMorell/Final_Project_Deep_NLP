from datasets import load_dataset
from textblob import TextBlob
import os

def download_data(path: str = "data"):
    """
    
    Args:
        path (str): path where the data is going to be saved
    """
    if not os.path.exists(path):
        os.makedirs(path)

    # download dataset
    ds = load_dataset("tner/ontonotes5")

    df_train = ds["train"].to_pandas()
    df_val = ds["validation"].to_pandas()
    df_test = ds["test"].to_pandas()

    # process datasets
    df_train = process_sentences(df_train)
    df_val = process_sentences(df_val)
    df_test = process_sentences(df_test)

    # save data in .csv
    df_train.to_csv(f"{path}/train.csv", index=False)
    df_val.to_csv(f"{path}/val.csv", index=False)
    df_test.to_csv(f"{path}/test.csv", index=False)

    return df_train, df_val, df_test

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < 0:
        return 0  # Negativo
    elif polarity == 0:
        return 1  # Neutro
    else:
        return 2  # Positivo
        
def process_sentences(df):
    # Crear la columna 'sentence' uniendo los tokens
    df["sentence"] = df["tokens"].apply(lambda x: " ".join(x))

    # Aplicar análisis de sentimiento
    df["SA"] = df["sentence"].apply(analyze_sentiment)
    
    return df

