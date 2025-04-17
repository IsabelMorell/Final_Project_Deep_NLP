# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Tuple

# own modules
from src.utils import Accuracy

def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss_ner: torch.nn.Module,
    loss_sa: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    train_accuracy_ner: Accuracy,
    train_accuracy_sa: Accuracy
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss_ner: loss function for NER.
        loss_sa: loss function for SA.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
        train_accuracy_ner: accuracy in train in NER.
        train_accuracy_sa: accuracy in train in SA.
    """
    model.train()

    # define metric lists
    losses_ner: list[float] = []
    losses_sa: list[float] = []

    for sentences, ner_labels, sentiment, text_len in train_data:
        sentences = sentences.to(device)
        ner_labels = ner_labels.to(device)
        sentiment = sentiment.to(device)

        outputs, sa = model(sentences, text_len)

        loss_value_ner = loss_ner(outputs, ner_labels[:,:text_len[0],:])
        loss_value_sa = loss_sa(sa, sentiment)

        loss_value = loss_value_ner + loss_value_sa

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        losses_ner.append(loss_value_ner.item())
        losses_sa.append(loss_value_sa.item())

        train_accuracy_ner.update(outputs, ner_labels)
        train_accuracy_sa.update(sa, sentiment)

    # write on tensorboard
    writer.add_scalar("train_ner/loss", np.mean(losses_ner), epoch)
    writer.add_scalar("train_sa/loss", np.mean(losses_sa), epoch)
    writer.add_scalar("train_ner/accuracy", train_accuracy_ner.compute(), epoch)
    writer.add_scalar("train_sa/accuracy", train_accuracy_sa.compute(), epoch)

def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss_ner: torch.nn.Module,
    loss_sa: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    val_accuracy_ner: Accuracy,
    val_accuracy_sa: Accuracy
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss_ner: loss function for NER.
        loss_sa: loss function for SA.
        scheduler: scheduler.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
        val_accuracy_ner: accuracy in validation in NER.
        val_accuracy_sa: accuracy in validation in SA.
    """
    model.eval()

    with torch.no_grad():
        # define metric lists
        losses_ner: list[float] = []
        losses_sa: list[float] = []

        for sentences, ner_labels, sentiment, text_len in val_data:
            sentences = sentences.to(device)
            ner_labels = ner_labels.to(device)
            sentiment = sentiment.to(device)

            outputs, sa = model(sentences, text_len)

            loss_value_ner = loss_ner(outputs, ner_labels)
            loss_value_sa = loss_sa(sa, sentiment)

            losses_ner.append(loss_value_ner.item())
            losses_sa.append(loss_value_sa.item())

            val_accuracy_ner.update(outputs, ner_labels)
            val_accuracy_sa.update(sa, sentiment)
        
        scheduler.step()

        # write on tensorboard
        writer.add_scalar("val_ner/loss", np.mean(losses_ner), epoch)
        writer.add_scalar("val_sa/loss", np.mean(losses_sa), epoch)
        writer.add_scalar("val_ner/accuracy", val_accuracy_ner.compute(), epoch)
        writer.add_scalar("val_sa/accuracy", val_accuracy_sa.compute(), epoch)

def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
    test_accuracy_ner: Accuracy,
    test_accuracy_sa: Accuracy,
) -> Tuple[float, float]:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        test_data: dataloader of test data.
        device: device of model.
        test_accuracy_ner: accuracy in test in NER.
        test_accuracy_sa: accuracy in test in SA.

    Returns:
        accuracy_ner (float): average accuracy in NER
        accuracy_sa (float): average accuracy in SA
    """
    model.eval()

    with torch.no_grad():
        for sentences, ner_labels, sentiment, text_len in test_data:
            sentences = sentences.to(device)
            ner_labels = ner_labels.to(device)
            sentiment = sentiment.to(device)

            outputs, sa = model(sentences, text_len)

            test_accuracy_ner.update(outputs, ner_labels)
            test_accuracy_sa.update(sa, sentiment)

    # write on tensorboard
    return test_accuracy_ner.compute(), test_accuracy_sa.compute()