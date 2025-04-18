# deep learning libraries
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from torch.jit import RecursiveScriptModule
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

# other libraries
import os
import random
import spacy
import re
import ast

# own modules
from src.data import ENTITY2INDEX, SA2INDEX, NUM_NER_CLASSES, NUM_SA_CLASSES

CONTRACTIONS = {
    "n't": "not", "'ll": "will", "'re": "are", "'ve": "have", "'m": "am", 
    "'d": "would", "'s": "is", "won't": "will not", "can't": "cannot"
}
IRRELEVANT_WORDS = {"wow", "oops", "ah", "ugh", "yay", "mhm", "`"}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# HOLA SOY MARÍA, AQUÍ NO SE QUE PONER EN LO QUE DEVUELVE
def fix_tags_string(x) -> List[int]:
    """
    This function processes a string of comma-separated numbers and converts it into a list 
    of integers. If the input is not a string, it returns the input unchanged.

    Args:
        x: a string containing comma-separated numbers.

    Returns:
        A list of integers if x is a string, otherwise returns the original input unchanged.
    """
    if isinstance(x, str):
        x_clean = re.sub(r"\s+", ",", x.strip())
        nums = [int(n) for n in x_clean.strip("[]").split(",") if n.strip() != ""]
        return nums
    return x 

def replace_contractions(text) -> str:
    """
    This function replaces the contractions present in the given text with their 
    expanded form

    Args:
        text: string containing the sentence with contractions.

    Returns:
        string containing the text without contractions, replaced by their full forms.
    """
    for contraction, replacement in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", replacement, text)
    return text

def process_sentence_and_align_tags(sentence, original_tags) -> Tuple[List, List]:
    """
    This function processes a sentence by removing contractions, punctuation, stop words,
    and other irrelevant words. Then, aligns the given tags with the filtered tokens.

    Args:
        sentence: string with the raw input
        original_tags: a list of tags corresponding to the original tokens in the sentence.

    Returns:
        Tuple containing:
            - processed_tokens: a list of cleaned, lemmatized tokens.
            - aligned_tags: a list of tags aligned with the processed tokens.
    """
    nlp = spacy.load("en_core_web_sm")

    sentence = replace_contractions(sentence)
    sentence = sentence.replace("-", "")
    doc = nlp(sentence)

    processed_tokens = []
    aligned_tags = []

    tag_idx = 0
    for token in doc:
        if token.is_punct or token.is_space or token.text.lower() in IRRELEVANT_WORDS:
            tag_idx += 1  # Skip both token and its tag
            continue
        if token.is_stop:
            tag_idx += 1
            continue

        processed_tokens.append(token.lemma_)

        if tag_idx < len(original_tags):
            aligned_tags.append(original_tags[tag_idx])
            tag_idx += 1
        else:
            aligned_tags.append(0)

    return processed_tokens, aligned_tags

def word2idx(embedding_dict, tweet) -> torch.Tensor:
    """
    This function converts a list of words (a tweet) into a tensor of corresponding indices
    using a given embedding dictionary.

    Args:
        embedding_dict: a dictionary mapping words to their corresponding indices.
        tweet: a list of words (tokens) from a tweet.

    Returns:
        tensor containing the indices of the words in the tweet

    """
    indices = [embedding_dict[word] for word in tweet if word in embedding_dict]
    if not indices:
        indices = [0]  
    return torch.tensor(indices)

def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function processes a batch of samples for a DataLoader, preparing text sequences
    and labels with padding and sorting by length.

    Args:
        batch: list of tuples where each tuple contains (text, label, sa)
            - text: list of words (tokens)
            - label: tensor with label indices
            - sa: auxiliary information

    Returns:
        tuple containing:
            - text_padded: padded tensor of word indices
            - tags_padded: padded tensor of label indices
            - sa: auxiliary information
            - lengths: tensor of original sequence lengths.

    """
    
    glove = spacy.load('en_core_web_lg')
    word_to_index = {word: i for i, word in enumerate(glove.vocab.strings)} 

    # Ordenar por longitud de la secuencia (descendente)
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    texts, labels, sa = zip(*batch)

    # Convertir palabras a índices
    # texts_indx = [word2idx(word_to_index, text) for text in texts if word2idx(word_to_index, text).nelement() > 0]
    
    texts_indx = []
    labels_indx = []
    for i, text in enumerate(texts):
        word_2_idx = word2idx(word_to_index, text)
        if word_2_idx.nelement() > 0:
            texts_indx.append(word_2_idx)
            labels_indx.append(labels[i])

    # Padding a la misma longitud
    texts_padded = pad_sequence(texts_indx, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(labels_indx, batch_first=True, padding_value=ENTITY2INDEX["O"]).long()

    # Longitudes de cada secuencia
    lengths = torch.tensor([len(text) for text in texts_padded], dtype=torch.long)
    
    # One hot NER
    batch_size, max_len = tags_padded.shape
    tags_onehot = torch.zeros((batch_size, int(lengths[0].item()), NUM_NER_CLASSES), dtype=torch.float)

    label_idx: int
    for i in range(batch_size):
        for j in range(lengths[i]):
            label_idx = int(tags_padded[i, j].item())
            tags_onehot[i, j, label_idx] = 1.0
        
    # One hot SA
    batch_size = len(sa)
    sa_onehot = torch.zeros((batch_size, NUM_SA_CLASSES), dtype=torch.float)

    for i in range(batch_size):
        label_idx = int(sa[i].item())
        sa_onehot[i, label_idx] = 1.0

    return texts_padded, tags_onehot, sa_onehot, lengths

def return_index2label(label2index: Dict[str, int]) -> Dict[int, str]:
    index2label = {}
    for label in label2index:
        index = label2index[label]
        index2label[index] = label
    return index2label

class Accuracy:
    """
    This class is the accuracy object.

    Attr:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self, task: Optional[str] = None) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0
        self.task = task
        
        if task:
            if task.lower() == "ner":
                idx2entity: Dict[int, str] = return_index2label(ENTITY2INDEX)
                self.idxs: List[int] = list(idx2entity.keys())
                self.correct_ner_occurrences = {idx: 0 for idx in idx2entity.keys()}
                self.ner_occurrences = {idx: 0 for idx in idx2entity.keys()}
            elif task.lower() == "sa":
                pass
            else:
                raise ValueError("Please introduce a valid task like ner or sa")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits (torch.Tensor): outputs of the model.
                Dimensions: [batch, number of classes]

            labels: labels of the examples. Dimensions: [batch].
        """

        # compute predictions
        predictions = logits.argmax(-1).type_as(labels)
        labels = labels.argmax(-1).type_as(labels)

        # update counts
        self.correct += int(predictions.eq(labels).sum().item())        
        self.total += labels.numel()
        
        if self.task == "ner":
            for idx in self.idxs:
                labels_idx = (labels.to(torch.long) == idx)
                predictions_idx = (predictions.to(torch.long) == idx)
                    
                correct_ner_idx = (labels_idx == predictions_idx)*labels_idx
                
                self.correct_ner_occurrences[idx] += int(correct_ner_idx.sum().item())  
                
                self.ner_occurrences[idx] += int((labels_idx.sum()).item())

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            accuracy value.
        """

        return self.correct / self.total
    
    def ner_entities_accuracy(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.task == "ner":
            return self.correct_ner_occurrences, self.ner_occurrences
        else:
            raise ValueError("This function is only avaliable for ner tasks")

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # init to zero the counts
        self.correct = 0
        self.total = 0

        return None

def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model (torch.nn.Module): pytorch model.
        name (str): name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name (str): name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed (int): seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
