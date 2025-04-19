
# Deep Learning + NLP Final Project

The objective of this final project for the Natural Language Processing I and Deep Learning courses is to design an alert generation system based on news articles and social media posts. The system generates alerts according to the predicted Named Entity Recognition (NER) tags and the sentiment of the text (SA). We have developed a model that performs both tasks jointly: NER to identify entities and SA to classify the sentiment. Finally, we adapt a prompt with the model's predictions to ask Ollama to generate the final alert, completing the Alert Generation System.

# Important Notes to Run the Code

This project depends on some SpaCy models, such as `en_core_web_sm` and `en_core_web_lg`. If you haven't installed them yet, or if you encounter one of the following errors:

    OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

    OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.

Run the following commands:

    python -m spacy download en_core_web_sm

    python -m spacy download en_core_web_lg

# Structure of the Repository

- All code related to the **NER + SA model** is located in the src/ folder.

- Code for the **alert generation system** is located in the alert_generation/ folder.

## Training the model

To train the NER + SA model and save it, run:

    python -m src.train

In the `src.train` module, there are some hyperparameters defined at the beginning of the main function that can be modified to optimize performance.

## Evaluating the model

To evaluate the model, run:

    python -m src.evaluate

This script loads the `best_model.pt` file from the models/ folder, computes the accuracy on the test set, and prints the result to the command line.

## Training our model

To train our NER + SA model you must use as hyperparameters 30 epoch, a step size of 20, a learning rate of \( 1 \times 10^{-3} \), a weight decay of \( 1 \times 10^{-3} \), a batch size of 128, a hidden size of 256, two layers and a dropout rate of 0.3. This hyperparameters are found in the `src/train.py` file on line 43.

To train the NER + SA model and save it, run:

    python -m src.train

## Generating alerts

To generate alerts you need to download a ollama model like llama3.3. For this you need to run the following comand replacing llama3.3 with the name of the model you want:

    ollama pull llama3.3

Note that you need to have installed the ollama library specified in the requirements.txt

To know the models available in ollama check out the following link: https://ollama.com/library

To generate alerts based on text input, run:

    python -m alert_generation.generate_alerts

In the `alert_generation.generate_alerts` module, the `MODEL` variable at the beginning of the script can be modified to choose the LLM model used for generating alerts. You can also modify the `sentences` list to test the alert generation system with custom sentences.

# Running the Code with Your Own Data

If you want to train the model with your own dataset:

1. Place your .csv files in a data/ folder (create it if it does not exist) and name them:
    - `train.csv`
    - `val.csv`
    - `test.csv`

2. If you have old files named `train_token.csv`, `val_token.csv`, or `test_token.csv` in your data/ folder, delete them if they were not generated with your new data.

3. Then, simply run:

    python -m src.train

The training script will automatically clean your data and generate the `_token.csv` files needed for training.

## Running the complete pipeline (NER + SA + Alert Generation)
If you want to test our the whole pipeline with out pre-trained NERSA model, then follow the next steps:

1. Download the ollama model you want to use

2. In the file `alert_generation/generate_alerts.py`, modify the variable `MODEL` (line 19) with the
name if the model you have dowloaded

3. Write the sentences you want to analyze in the variable `sentences` (line 155) in the same file (`alert_generation/generate_alerts.py`)
