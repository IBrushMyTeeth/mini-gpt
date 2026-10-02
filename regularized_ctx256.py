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
LEARNING_CURVE_PATH = ROOT / "plots" / "regularized_ctx256_learning_curve.png"
MODEL_PATH = ROOT / "check_points" / "regularized_ctx256.pt"


def main():

    data = torch.load(DATA_PATH)
    vocabulary_size = data["vocabulary_size"]

    train_tokens = data["train_split"]
    validation_tokens = data["validation_split"]

    train_dataset = ShakespeareDataset(
        token_ids=train_tokens,
        context_length=256,
        stride=128,
    )

    validation_dataset = ShakespeareDataset(
        token_ids=validation_tokens,
        context_length=256,
        stride=256,
    )

    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, 64, shuffle=False)

    model_config = ModelConfig(
        vocabulary_size=vocabulary_size,
        max_sequence_length=256,
    )

    model = GPT(model_config, use_dropout=True)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=0.1
    )

    criterion = nn.CrossEntropyLoss()

    history_dict = train(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=30,
        validation_loader=validation_loader,
    )

    save_learning_curves(history_dict, LEARNING_CURVE_PATH)
    model.save(MODEL_PATH)

    
if __name__ == "__main__":
    main()


"""
Initializing training:
Currently at epoch 1...
Training loss: 2.6206 | Validation loss: 2.4589
Currently at epoch 2...
Training loss: 2.3600 | Validation loss: 2.2456
Currently at epoch 3...
Training loss: 2.0862 | Validation loss: 1.9865
Currently at epoch 4...
Training loss: 1.8838 | Validation loss: 1.8664
Currently at epoch 5...
Training loss: 1.7565 | Validation loss: 1.7715
Currently at epoch 6...
Training loss: 1.6694 | Validation loss: 1.7112
Currently at epoch 7...
Training loss: 1.6053 | Validation loss: 1.6582
Currently at epoch 8...
Training loss: 1.5574 | Validation loss: 1.6242
Currently at epoch 9...
Training loss: 1.5211 | Validation loss: 1.6009
Currently at epoch 10...
Training loss: 1.4904 | Validation loss: 1.5717
Currently at epoch 11...
Training loss: 1.4635 | Validation loss: 1.5606
Currently at epoch 12...
Training loss: 1.4415 | Validation loss: 1.5442
Currently at epoch 13...
Training loss: 1.4224 | Validation loss: 1.5314
Currently at epoch 14...
Training loss: 1.4044 | Validation loss: 1.5202
Currently at epoch 15...
Training loss: 1.3892 | Validation loss: 1.5172
Currently at epoch 16...
Training loss: 1.3754 | Validation loss: 1.5091
Currently at epoch 17...
Training loss: 1.3631 | Validation loss: 1.5051
Currently at epoch 18...
Training loss: 1.3509 | Validation loss: 1.5063
Currently at epoch 19...
Training loss: 1.3402 | Validation loss: 1.5015
Currently at epoch 20...
Training loss: 1.3298 | Validation loss: 1.4985
Currently at epoch 21...
Training loss: 1.3199 | Validation loss: 1.4975
Currently at epoch 22...
Training loss: 1.3112 | Validation loss: 1.4985
Currently at epoch 23...
Training loss: 1.3027 | Validation loss: 1.4943
Currently at epoch 24...
Training loss: 1.2947 | Validation loss: 1.4913
Currently at epoch 25...
Training loss: 1.2875 | Validation loss: 1.4951
Currently at epoch 26...
Training loss: 1.2797 | Validation loss: 1.4950
Currently at epoch 27...
Training loss: 1.2727 | Validation loss: 1.4917
Currently at epoch 28...
Training loss: 1.2661 | Validation loss: 1.4930
Currently at epoch 29...
Training loss: 1.2590 | Validation loss: 1.4956
Currently at epoch 30...
Training loss: 1.2530 | Validation loss: 1.5006
Best epoch: 24 | Lowest validation loss: 1.4913


Loss comparison for the model trained with context length 256:

The 256-context model showed a steady decrease in both training and validation
loss throughout most of training. The training loss continued to improve up
to epoch 30, reaching 1.2530, while the validation loss reached its lowest
value of 1.4913 at epoch 24.

After epoch 24, the validation loss remained relatively stable but began to
increase slightly, indicating that further training was no longer improving
generalization. The best checkpoint was therefore obtained at epoch 24. This
was however above the best achieved validation loss for the previous model
which had a context length of 128 and a validation loss of 1.4736.
The results also indicate that the model had largely reached a validation-loss
plateau by the later stages of training.
"""