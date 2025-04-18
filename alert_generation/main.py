# deep learning libraries
import torch
from torch.jit import RecursiveScriptModule

# other libraries
import os
from typing import Dict

# own modules
import utils as u
from src.utils import set_seed, load_model, return_index2label
from src.data import ENTITY2INDEX, SA2INDEX

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

placeholder = [
    "[S]",  # sentiment of the sentence
    "[N]"  # Sentence + NER tags
]

def sustituir_prompt(prompt: str, sentence_ner: str, sa: str, placeholders: list) -> str:
    """
    Reemplaza los placeholders en el prompt con el contenido de los archivos en paths.
    
    prompt: str: Enunciado con los placeholders [E], [P], [R], [C], [A]
    paths: list: Lista de rutas a los archivos [Enunciado, Código Profesor, Rubrica, Correcion, Código Alumno]
    
    Returns:
    str: El prompt con los placeholders reemplazados por el contenido de los archivos.
    """

    # Loop through the placeholders and replace them in the prompt
    for i in range(len(placeholders)):
        if placeholder[i] == "[S]":
            content = sa
        else:
            content = sentence_ner
        
        # Replace the placeholder with the corresponding content
        prompt = prompt.replace(placeholders[i], content)
    
    return prompt

def abrir_y_ejecutar_prompt(sentence_ner: str, sa: str):
    # Leer el contenido del archivo 'prompt.txt' que está en la carpeta actual
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()
    except FileNotFoundError:
        print("El archivo 'prompt.txt' no se encuentra en la carpeta actual.")
        return
    except Exception as e:
        print(f"Error al leer el archivo 'prompt.txt': {e}")
        return
    
    prompt_sustituido = sustituir_prompt(prompt, sentence_ner, sa, placeholder)
    
    # Obtener los modelos disponibles
    models = u.get_available_models()[0:-4]
    if not models:
        print("No se encontraron modelos disponibles.")
        return
    
    # Para cada modelo, ejecutar la función 'prompt_model' pasando el prompt sustituido
    for model in models:
        u.delete_history()
        print(f"Ejecutando para el modelo: {model}")
        try:
            response = u.prompt_model(model, prompt_sustituido, path=".")
            return response
        except Exception as e:
            print(f"Error al ejecutar para el modelo {model}: {e}")

def most_probable_entity(logits: torch.Tensor, index2entity: dict) -> str:
    idx_most_probable: int = torch.argmax(logits).to(int).item()
    entity: str = index2entity[idx_most_probable]
    return entity

def add_ner_to_sentence(sentence: str, ner_tags: list) -> str:
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
    
    # 1. Pasar frase original por el modelo
    ner_logits, sa_logits = model()

    # 2. Sacar NER mas probable por palabra y recuperar la tag asociada al indice
    ner_tags: list = [most_probable_entity(ner_logits[i, :], INDEX2ENTITY) for i, word in enumerate(sentence.split(" "))]

    # 3. Sacar SA mas probable y su tag asociado
    # sa = "negative"
    # sa = "positive"
    sa: str = most_probable_entity(sa_logits, INDEX2SA)

    # 4. Funcion que añade las etiquetas a la frase
    # sentence_ner = "Child (B-PER) murdered (O) in (O) Florida (B-LOC)"
    # sentence_ner = "Gay (B-EVENT) marriage (I-EVENT) has (O) been (O) legalized (O) in (O) England (B-LOC)!"
    sentence_ner = add_ner_to_sentence(sentence, ner_tags)

    try:
        response = abrir_y_ejecutar_prompt(sentence_ner, sa)
        print("RESPONSE")
        print(response)
    except Exception as error:
        print("ERROR:", error)