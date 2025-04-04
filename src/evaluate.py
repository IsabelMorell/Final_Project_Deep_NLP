# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.utils import Accuracy, set_seed, load_model
from src.train_functions import t_step

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

# static variables
DATA_PATH: Final[str] = "data"

def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    batch_size: int = 128
    # load data
    _, _, test_data = load_data(DATA_PATH, batch_size=batch_size)

    # define model
    model: RecursiveScriptModule = load_model(name)

    # call test step and evaluate accuracy
    test_accuracy_ner: Accuracy = Accuracy()
    test_accuracy_sa: Accuracy = Accuracy()
    accuracy_ner: float 
    accuracy_sa: float
    accuracy_ner, accuracy_sa = t_step(model, test_data, device, test_accuracy_ner, test_accuracy_sa)

    return accuracy_ner, accuracy_sa


if __name__ == "__main__":
    accuracy_ner, accuracy_sa = main('best_model')
    print(f"accuracy_ner: {accuracy_ner}")
    print(f"accuracy_sa: {accuracy_sa}")
