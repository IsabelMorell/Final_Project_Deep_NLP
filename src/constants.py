import spacy

# Processing and embedding models
NLP = spacy.load("en_core_web_sm")
GLOVE = spacy.load("en_core_web_lg")

# Constants used to clean and process data
CONTRACTIONS = {
    "n't": "not",
    "'ll": "will",
    "'re": "are",
    "'ve": "have",
    "'m": "am",
    "'d": "would",
    "'s": "is",
    "won't": "will not",
    "can't": "cannot",
}
IRRELEVANT_WORDS = {"wow", "oops", "ah", "ugh", "yay", "mhm", "`"}

# label to index dictionaries
SA2INDEX = {"negative": 0, "neutral": 1, "positive": 2}

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
    "I-LANGUAGE": 36,
}

NUM_NER_CLASSES = len(ENTITY2INDEX)
NUM_SA_CLASSES = len(SA2INDEX)
