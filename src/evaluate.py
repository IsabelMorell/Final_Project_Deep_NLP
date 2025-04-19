# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final, Tuple

# own modules
from src.data import load_data
from src.utils import ENTITY2INDEX, Accuracy, set_seed, load_model, return_index2label
from src.train_functions import t_step

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

# static variables
DATA_PATH: Final[str] = "data"

def evaluate(name: str) -> Tuple[float, float, Accuracy]:
    """
    This function is the main program for the testing.
    """

    batch_size: int = 128
    # load data
    test_data: DataLoader
    _, _, test_data = load_data(batch_size=batch_size)

    # define model
    model: RecursiveScriptModule = load_model(name)
    model.to(device)

    # call test step and evaluate accuracy
    test_accuracy_ner: Accuracy = Accuracy(task="ner")
    test_accuracy_sa: Accuracy = Accuracy()
    accuracy_ner: float 
    accuracy_sa: float
    accuracy_ner, accuracy_sa = t_step(model, test_data, device, test_accuracy_ner, test_accuracy_sa)

    return accuracy_ner, accuracy_sa, test_accuracy_ner


if __name__ == "__main__":
    accuracy_ner: float
    accuracy_sa: float
    accuracy_ner, accuracy_sa, test_accuracy_ner = evaluate('best_model')
    print(f"accuracy_ner: {accuracy_ner}")
    print(f"accuracy_sa: {accuracy_sa}")
    
    correct_ner_occurrences, ner_occurrences = test_accuracy_ner.ner_entities_accuracy()
    idx2entity = return_index2label(ENTITY2INDEX)
    for idx, entity in idx2entity.items():
        print(f"Entity {entity}: {correct_ner_occurrences[idx]}/{ner_occurrences[idx]}")