# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# other libraries
import os
from typing import List, Dict, Optional

# own modules
import utils as u
from src.utils import set_seed, load_model, return_index2label
from src.utils import ENTITY2INDEX, SA2INDEX

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

placeholder: List[str] = [
    "[S]",  # sentiment of the sentence
    "[N]"  # sentence + NER tags
]

def replace_prompt(prompt: str, sentence_ner: str, sa: str, placeholders: List[str]) -> str:
    """
    Replaces the placeholders in the prompt with the content of the sentence_ner and sa strings.
    
    Args:
        prompt (str): alert generation tasks with the placeholders [S], [N] where the first refers to the sentence's
            sentiment and the second to the sentence along with the NER tags associated to the words.
        sentence_ner (str): input sentence with its associated NER tags.
        sa (str): sentiment of the input sentence.
        placeholders (list): placeholders to replace
    
    Returns:
        prompt (str): prompt with the placeholders replaced by the sa and sentence_ner strings.
    """
    
    content: str
    # Loop through the placeholders and replace them in the prompt
    for i in range(len(placeholders)):
        if placeholder[i] == "[S]":
            content = sa
        else:
            content = sentence_ner
        
        # Replace the placeholder with the corresponding content
        prompt = prompt.replace(placeholders[i], content)
    
    return prompt

def abrir_y_ejecutar_prompt(sentence_ner: str, sa: str) -> Optional[str]:
    """
    Reads the content of the file prompt.txt, replaces the placeholders and 
    prompts the model.

    Args:
        sentence_ner (str): input sentence with its associated NER tags.
        sa (str): sentiment of the input sentence.

    Returns:
        response (str): models response to the prompt
    """
    
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            prompt: str = file.read()
    except FileNotFoundError:
        print("The file 'prompt.txt' isn't in the current folder.")
        return
    except Exception as e:
        print(f"Error when reading file 'prompt.txt': {e}")
        return
    
    replaced_prompt: str = replace_prompt(prompt, sentence_ner, sa, placeholder)
    
    # Get available models
    models = u.get_available_models()[0:-4]
    if not models:
        print("There are no models available.")
        return
    
    # For each model, run function 'prompt_model' with the replaced prompt
    for model in models:
        u.delete_history()
        print(f"Running for model: {model}")
        try:
            response = u.prompt_model(model, replaced_prompt, path=".")
            return response
        except Exception as e:
            print(f"Error while running model {model}: {e}")

def most_probable_entity(logits: torch.Tensor, index2entity: Dict[int, str]) -> str:
    """
    Return the entity with the highest probability

    Args:
        logits (torch.Tensor): probabilities associated with a specific index
        index2entity (Dict[int, str]): dictionary with the relationship between the indexes and the entities

    Returns:
        entity (str): entity associated with the highest probability
    """
    
    idx_most_probable: int = torch.argmax(logits).to(int).item()
    entity: str = index2entity[idx_most_probable]
    return entity

def add_ner_to_sentence(sentence: str, ner_tags: List[str]) -> str:
    """
    For every word in a sentence, it is followed by its associated NER tag written in parenthesis.

    Args:
        sentence (str): input sentence
        ner_tags (List[str]): NER tags associated to each word of the input sentence

    Returns:
        ner_sentence (str): sentence where every word is followed by its associated NER tag written in parenthesis
    """
    ner_sentence = ""
    for i, word in enumerate(sentence.split(" ")):
        ner_sentence += f"{word} ({ner_tags[i]})"
        if i != (len(ner_tags) - 1):
            ner_sentence += " "

    return ner_sentence


if __name__ == "__main__":
    INDEX2ENTITY = return_index2label(ENTITY2INDEX)
    INDEX2SA = return_index2label(SA2INDEX)
    sentence = "Child murdered in Florida"
    
    # Load the model
    model: RecursiveScriptModule = load_model("best_model")
    model.to(device)
    
    # 1. Pass original sentence through the model
    ner_logits, sa_logits = model()

    # 2. Obtain most probable NER tag for each word
    ner_tags: list = [most_probable_entity(ner_logits[i, :], INDEX2ENTITY) for i, word in enumerate(sentence.split(" "))]

    # 3. Obtain most probable sentence sentiment
    # sa = "negative"
    # sa = "positive"
    sa: str = most_probable_entity(sa_logits, INDEX2SA)

    # 4. Add NER tags to the sentence
    # sentence_ner = "Child (B-PER) murdered (O) in (O) Florida (B-LOC)"
    # sentence_ner = "Gay (B-EVENT) marriage (I-EVENT) has (O) been (O) legalized (O) in (O) England (B-LOC)!"
    sentence_ner = add_ner_to_sentence(sentence, ner_tags)

    try:
        response = abrir_y_ejecutar_prompt(sentence_ner, sa)
        print("RESPONSE")
        print(response)
    except Exception as error:
        print("ERROR:", error)