# deep learning libraries
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import numpy as np
import torch
import ast
import pandas as pd
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader

# other libraries
import os

from utils import fix_tags_string, process_sentence_and_align_tags, collate_fn

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
ENTITY2INDEX = {
    "O": 0,
    "B-CARDINAL": 1,
    "B-DATE": 2,
    "I-DATE": 3,
    "B-PERSON": 4,
    "I-PERSON": 5,
    "B-NORP": 6,
    "B-GPE": 7,
    "I-GPE": 8,
    "B-LAW": 9,
    "I-LAW": 10,
    "B-ORG": 11,
    "I-ORG": 12, 
    "B-PERCENT": 13,
    "I-PERCENT": 14, 
    "B-ORDINAL": 15, 
    "B-MONEY": 16, 
    "I-MONEY": 17, 
    "B-WORK_OF_ART": 18, 
    "I-WORK_OF_ART": 19, 
    "B-FAC": 20, 
    "B-TIME": 21, 
    "I-CARDINAL": 22, 
    "B-LOC": 23, 
    "B-QUANTITY": 24, 
    "I-QUANTITY": 25, 
    "I-NORP": 26, 
    "I-LOC": 27, 
    "B-PRODUCT": 28, 
    "I-TIME": 29, 
    "B-EVENT": 30,
    "I-EVENT": 31,
    "I-FAC": 32,
    "B-LANGUAGE": 33,
    "I-PRODUCT": 34,
    "I-ORDINAL": 35,
    "I-LANGUAGE": 36
}

class OntoNotesDataset(Dataset):
    """

    This class is a custom PyTorch Dataset for the OntoNotes dataset.

    Args:
        df: A pandas DataFrame containing at least the columns 'tokens', 'tags', and 'SA'.

    """
    def __init__(self, df: pd.DataFrame):

        """
        Initializes the dataset

        Args:
            df: A DataFrame containing 'tokens', 'tags', and 'SA' columns.

        """
        self.tokens = df["tokens"].tolist()

        df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.tags = [torch.tensor(t, dtype=torch.float32) for t in df["tags"].tolist()]
        
        self.SA: torch.Tensor = torch.tensor(df["SA"].values, dtype=torch.int)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (number of samples).
        
        Returns:
            The number of samples (tokens) in the dataset.
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Retrieves the token sequence, associated tags, and sentiment label for a given index.
        
        Args:
            idx: The index of the sample to retrieve.
        
        Returns:
            A tuple containing:
                - token: The list of tokens at the given index.
                - tag: The tensor of tags associated with the tokens.
                - sa: The sentiment analysis label for the sample.
        """
        token: str = self.tokens[idx]
        tag: torch.Tensor = self.tags[idx]
        sa: torch.Tensor = self.SA[idx]
        return token, tag, sa

def download_data(path: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function downloads the OntoNotes dataset, processes it, and saves the resulting data
    as CSV files. The dataset is split into train, validation, and test sets.

    Args:
        path (str): The directory path where the data will be saved. Defaults to "data".
        
    Returns:
        A tuple containing three DataFrames:
            - df_train: The processed training data.
            - df_val: The processed validation data.
            - df_test: The processed test data.
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

# Preprocess text (username and link placeholders)
def preprocess(text:str) -> str:
    """
    This function processes a given text by replacing usernames (starting with '@') with 
    the placeholder '@user', and URLs (starting with 'http') with the placeholder 'http'.
    
    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text with usernames and links replaced by placeholders.
    """
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(sentence:str) -> int:
    """
    This function performs sentiment analysis on a given sentence by preprocessing it, 
    tokenizing it, passing it through a pre-trained model, and returning the index of 
    the highest sentiment score.

    Args:
        sentence (str): The input sentence to analyze.

    Returns:
        int: The index of the highest sentiment score, indicating the sentiment classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    sentence = preprocess(sentence)
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return ranking[0]
        
def process_sentences(df:pd.DataFrame) -> pd.DataFrame:
    """
    This function processes a DataFrame by generating a 'sentence' column from the 'tokens' column 
    and applies sentiment analysis to each sentence to create a new 'SA' column with sentiment labels.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'tokens' column.

    Returns:
        pd.DataFrame: The processed DataFrame with new 'sentence' and 'SA' columns.
    """

    # Crear la columna 'sentence' uniendo los tokens
    df["sentence"] = df["tokens"].apply(lambda x: " ".join(x))

    # Aplicar análisis de sentimiento
    df["SA"] = df["sentence"].apply(analyze_sentiment)
    
    return df


def load_data(batch_size: int=64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function loads and processes the OntoNotes dataset. It downloads the data if it's not 
    available locally, preprocesses it, and returns PyTorch DataLoader objects for training, validation, 
    and testing datasets.

    Args:
        batch_size (int): The batch size to use for the DataLoader. Defaults to 64.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the DataLoader for training, 
        validation, and testing datasets.
    """
    if not os.path.exists("data"):
        df_train, df_val, df_test = download_data()
    else:
        df_train = pd.read_csv("data/train.csv")
        df_val = pd.read_csv("data/val.csv")
        df_test = pd.read_csv("data/test.csv")
    
    if "test_token.csv" not in os.listdir("data"):
        df_train["tags"] = df_train["tags"].apply(fix_tags_string)
        df_val["tags"] = df_val["tags"].apply(fix_tags_string)
        df_test["tags"] = df_test["tags"].apply(fix_tags_string)


        df_train[["tokens", "tags"]] = df_train.apply(
            lambda row: pd.Series(process_sentence_and_align_tags(row["sentence"], row["tags"])), axis=1
        )
        df_val[["tokens", "tags"]] = df_val.apply(
            lambda row: pd.Series(process_sentence_and_align_tags(row["sentence"], row["tags"])), axis=1
        )
        df_test[["tokens", "tags"]] = df_test.apply(
            lambda row: pd.Series(process_sentence_and_align_tags(row["sentence"], row["tags"])), axis=1
        )

        df_train.to_csv(f"data/train_token.csv",index=False)
        df_test.to_csv(f"data/test_token.csv",index=False)
        df_val.to_csv(f"data/val_token.csv",index=False)
    
    else:
        df_train = pd.read_csv("data/train_token.csv")
        df_val = pd.read_csv("data/val_token.csv")
        df_test = pd.read_csv("data/test_token.csv")

        df_train["tokens"] = df_train["tokens"].apply(ast.literal_eval)
        df_train["tags"] = df_train["tags"].apply(ast.literal_eval)

        df_val["tokens"] = df_val["tokens"].apply(ast.literal_eval)
        df_val["tags"] = df_val["tags"].apply(ast.literal_eval)

        df_test["tokens"] = df_test["tokens"].apply(ast.literal_eval)
        df_test["tags"] = df_test["tags"].apply(ast.literal_eval)

    tr_dataset = OntoNotesDataset(df_train)
    vl_dataset = OntoNotesDataset(df_val)
    ts_dataset = OntoNotesDataset(df_test)

    train_dataloader: DataLoader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_dataloader: DataLoader = DataLoader(vl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_dataloader: DataLoader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


