import spacy
import re

CONTRACTIONS = {
    "n't": "not", 
    "'ll": "will", 
    "'re": "are", 
    "'ve": "have", 
    "'m": "am", 
    "'d": "would", 
    "'s": "is", 
    "won't": "will not", 
    "can't": "cannot"
}
IRRELEVANT_WORDS = {"wow", "oops", "ah", "ugh", "yay", "mhm", "`"}

def replace_contractions(text):
    for contraction, replacement in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", replacement, text)
    return text

def process_tokens(text):
    nlp = spacy.load("en_core_web_sm")
    text = replace_contractions(text)
    text = text.replace("-", "")
    # Crear un objeto Doc de spaCy para cada palabra
    doc = nlp(text)  # Unir las palabras en una cadena y procesarlas con spaCy
    
    # Filtrar tokens: eliminar puntuación, stopwords y lematizar
    processed_words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.lower() not in IRRELEVANT_WORDS and not token.is_digit]
    
    return processed_words