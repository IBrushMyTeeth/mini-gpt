from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ShakespeareDataset
from model.config import ModelConfig
from model.gpt import GPT
from training.utils import save_learning_curves, train

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "shakespeare_char_tokens.pt"
LEARNING_CURVE_PATH = ROOT / "plots" / "baseline_learning_curve.png"
MODEL_PATH = ROOT / "check_points" / "baseline.pt"


def main():

    data = torch.load(DATA_PATH)
    vocabulary_size = data["vocabulary_size"]

    train_tokens = data["train_split"]
    validation_tokens = data["validation_split"]

    train_dataset = ShakespeareDataset(
        token_ids=train_tokens,
        context_length=128,
        stride=64,
    )

    validation_dataset = ShakespeareDataset(
        token_ids=validation_tokens,
        context_length=128,
        stride=128,
    )

    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, 64, shuffle=False)

    model_config = ModelConfig(vocabulary_size)
    model = GPT(model_config)


    optimizer = torch.optim.AdamW(
        model.parameters(),
    )

    criterion = nn.CrossEntropyLoss()

    history_dict = train(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=20,
        validation_loader=validation_loader,
    )

    save_learning_curves(history_dict, LEARNING_CURVE_PATH)
    model.save(MODEL_PATH)

    
if __name__ == "__main__":
    main()