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


"""
Results from terminal:

baseline_model.py 
Initializing training:
Currently at epoch 1...
Training loss: 2.3689 | Validation loss: 2.0696
Currently at epoch 2...
Training loss: 1.7968 | Validation loss: 1.7949
Currently at epoch 3...
Training loss: 1.5821 | Validation loss: 1.6700
Currently at epoch 4...
Training loss: 1.4753 | Validation loss: 1.5971
Currently at epoch 5...
Training loss: 1.4056 | Validation loss: 1.5653
Currently at epoch 6...
Training loss: 1.3550 | Validation loss: 1.5418
Currently at epoch 7...
Training loss: 1.3159 | Validation loss: 1.5431
Currently at epoch 8...
Training loss: 1.2801 | Validation loss: 1.5390
Currently at epoch 9...
Training loss: 1.2481 | Validation loss: 1.5395
Currently at epoch 10...
Training loss: 1.2172 | Validation loss: 1.5573
Currently at epoch 11...
Training loss: 1.1898 | Validation loss: 1.5817
Currently at epoch 12...
Training loss: 1.1646 | Validation loss: 1.5987
Currently at epoch 13...
Training loss: 1.1363 | Validation loss: 1.6154
Currently at epoch 14...
Training loss: 1.1125 | Validation loss: 1.6406
Currently at epoch 15...
Training loss: 1.0859 | Validation loss: 1.6603
Currently at epoch 16...
Training loss: 1.0603 | Validation loss: 1.7053
Currently at epoch 17...
Training loss: 1.0374 | Validation loss: 1.7236
Currently at epoch 18...
Training loss: 1.0159 | Validation loss: 1.7669
Currently at epoch 19...
Training loss: 0.9961 | Validation loss: 1.7949
Currently at epoch 20...
Training loss: 0.9689 | Validation loss: 1.8381


****** Find visualization at plots/baseline_learning_curve.png ******

The baseline model was trained for 20 epochs using a context length of 128 and
a batch size of 64. The training and validation losses were recorded after
each epoch.

The training loss decreased consistently from 2.3689 at epoch 1 to 0.9689 at
epoch 20, showing that the model was successfully learning the training data.

The validation loss decreased from 2.0696 to a minimum of 1.5390 at epoch 8.
After epoch 8, the validation loss began to increase while the training loss
continued to decrease. This divergence indicates that the model was beginning
to overfit the training data.

The best validation performance was therefore achieved at epoch 8. By epoch 20,
the training loss had decreased to 0.9689, while the validation loss had
increased to 1.8381.

Overall, the baseline model successfully learns patterns from the training
data, but begins to overfit after approximately eight epochs. Epoch 8 provides
the best validation performance. Future experiments shall therefore not simply
save the last epoch as a checkpoint, but instead try to capture the model
producing best validation loss.
"""