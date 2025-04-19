# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final
import spacy
import torch
import numpy as np

# own modules
from src.data import load_data
from src.constants import ENTITY2INDEX
from src.models import NERSA
from src.train_functions import train_step, val_step
from src.utils import Accuracy, set_seed, save_model

# set device and seed
device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: Final[str] = "data"

glove = spacy.load('en_core_web_lg')

def main() -> None:
    """
    This function is the main program for the training.
    """
    num_NER_labels: int = len(ENTITY2INDEX)
    num_SA_labels: int = 3

    # hyperparameters
    epochs: int = 30
    step_size: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(batch_size=batch_size)

    # define embedding weights
    vectors = [glove.vocab[word].vector for word in glove.vocab.strings]
    vectors_np = np.array(vectors)
    embedding_weights = torch.tensor(vectors_np)

    # define name and writer
    name: str = f"model_lr_{lr}_hs_{hidden_size}_{batch_size}_{epochs}_{step_size}_{num_layers}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    model: torch.nn.Module = NERSA(embedding_weights=embedding_weights, hidden_size=hidden_size, num_layers=num_layers, num_NER_labels=num_NER_labels, num_SA_labels=num_SA_labels, dropout=dropout)
    model.to(device)

    # define losses, optimizer and scheduler
    loss_ner: torch.nn.Module = torch.nn.CrossEntropyLoss()
    loss_sa: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=step_size
    )

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_accuracy_ner: Accuracy = Accuracy()
        train_accuracy_sa: Accuracy = Accuracy()
        train_step(
            model,
            train_data,
            loss_ner,
            loss_sa,
            optimizer,
            writer,
            epoch,
            device,
            train_accuracy_ner,
            train_accuracy_sa
        )
        train_accuracy_ner.reset()
        train_accuracy_sa.reset()

        # call val step
        val_accuracy_ner: Accuracy = Accuracy()
        val_accuracy_sa: Accuracy = Accuracy()
        val_step(model, val_data, loss_ner, loss_sa, scheduler, writer, epoch, device, val_accuracy_ner, val_accuracy_sa)
        val_accuracy_ner.reset()
        val_accuracy_sa.reset()

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()