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
LEARNING_CURVE_PATH = ROOT / "plots" / "regularized_learning_curve.png"
MODEL_PATH = ROOT / "check_points" / "regularized.pt"


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
        epochs=20,
        validation_loader=validation_loader,
    )

    save_learning_curves(history_dict, LEARNING_CURVE_PATH)
    model.save(MODEL_PATH)

    
if __name__ == "__main__":
    main()

"""
Initializing training:
Currently at epoch 1...
Training loss: 2.4254 | Validation loss: 2.1608
Currently at epoch 2...
Training loss: 1.9223 | Validation loss: 1.8802
Currently at epoch 3...
Training loss: 1.7039 | Validation loss: 1.7320
Currently at epoch 4...
Training loss: 1.5973 | Validation loss: 1.6336
Currently at epoch 5...
Training loss: 1.5308 | Validation loss: 1.5916
Currently at epoch 6...
Training loss: 1.4860 | Validation loss: 1.5602
Currently at epoch 7...
Training loss: 1.4537 | Validation loss: 1.5450
Currently at epoch 8...
Training loss: 1.4269 | Validation loss: 1.5316
Currently at epoch 9...
Training loss: 1.4078 | Validation loss: 1.5145
Currently at epoch 10...
Training loss: 1.3877 | Validation loss: 1.5089
Currently at epoch 11...
Training loss: 1.3723 | Validation loss: 1.4976
Currently at epoch 12...
Training loss: 1.3596 | Validation loss: 1.4971
Currently at epoch 13...
Training loss: 1.3496 | Validation loss: 1.4869
Currently at epoch 14...
Training loss: 1.3363 | Validation loss: 1.4814
Currently at epoch 15...
Training loss: 1.3262 | Validation loss: 1.4877
Currently at epoch 16...
Training loss: 1.3183 | Validation loss: 1.4736
Currently at epoch 17...
Training loss: 1.3112 | Validation loss: 1.4753
Currently at epoch 18...
Training loss: 1.3009 | Validation loss: 1.4784
Currently at epoch 19...
Training loss: 1.2926 | Validation loss: 1.4772
Currently at epoch 20...
Training loss: 1.2858 | Validation loss: 1.4752


Best epoch: 16 | Lowest validation loss: 1.4736


The regularized model was trained for 20 epochs using a context length of 128
and a batch size of 64. Dropout with a rate of 0.1 was enabled and the AdamW
optimizer used a weight decay of 0.1. Training and validation losses were
recorded after each epoch.

Compared to the baseline model, the regularized model achieved a lower best
validation loss of 1.4736 compared to 1.5390. The best validation performance
also occurred later, at epoch 16 rather than epoch 8.

At epoch 20, the regularized model had a training loss of 1.2858 and a
validation loss of 1.4752. This gives a much smaller difference between
training and validation loss than the baseline model, which had a training
loss of 0.9689 and a validation loss of 1.8381 at the same epoch.

The increased weight decay and dropout therefore reduced the divergence
between training and validation performance and allowed the model to
generalize better to the validation data. The best model was saved from epoch
16, which produced the lowest validation loss.

Overall, the regularized model improves upon the baseline by achieving a
lower validation loss while exhibiting substantially less overfitting.
"""