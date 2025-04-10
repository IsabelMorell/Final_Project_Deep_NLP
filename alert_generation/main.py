import utils as u
import os
import torch

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

SA2INDEX = {"negative": 0, "neutral": 1, "positive": 2}

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
            response = u.prompt_model(model, prompt_sustituido, ".")
            return response
        except Exception as e:
            print(f"Error al ejecutar para el modelo {model}: {e}")

def return_index2label(label2index: dict) -> dict:
    index2label = {}
    for label in label2index:
        index = label2index[label]
        index2label[index] = label
    return index2label

def most_probable_entity(logits: torch.Tensor, index2entity: dict) -> str:
    idx_most_probable: int = torch.argmax(logits).to(int).item()
    entity: str = index2entity[idx_most_probable]
    return entity


if __name__ == "__main__":
    INDEX2ENTITY = return_index2label(ENTITY2INDEX)
    INDEX2SA = return_index2label(SA2INDEX)
    sentence = "hi"
    # 1. Pasar frase original por el modelo
    # ner_logits, sa_logits = model(...)

    # 2. Sacar NER mas probable por palabra y recuperar la tag asociada al indice
    ner_tags: list = [most_probable_entity(ner_logits[i, :], INDEX2ENTITY) for i, word in enumerate(sentence.split(" "))]

    # 3. Sacar SA mas probable y su tag asociado
    #sa = "negative"
    #sa = "positive"
    sa_tag: str = most_probable_entity(sa_logits, INDEX2SA)


    # 4. Funcion que añade las etiquetas a la frase

    #sentence_ner = "Child (B-PER) murdered (O) in (O) Florida (B-LOC)"
    sentence_ner = "Gay (B-EVENT) marriage (I-EVENT) has (O) been (O) legalized (O) in (O) England (B-LOC)!"

    try:
        response = abrir_y_ejecutar_prompt(sentence_ner, sa)
        print("RESPONSE")
        print(response)
    except Exception as error:
        print("ERROR:", error)