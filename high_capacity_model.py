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
LEARNING_CURVE_PATH = ROOT / "plots" / "high_capacity_model_learning_curve.png"
MODEL_PATH = ROOT / "check_points" / "high_capacity.pt"


def main():

    data = torch.load(DATA_PATH)
    vocabulary_size = data["vocabulary_size"]

    train_tokens = data["train_split"]
    validation_tokens = data["validation_split"]

    train_dataset = ShakespeareDataset(
        token_ids=train_tokens,
        context_length=128,
        # Use non-overlapping training windows to reduce CPU training time.
        stride=128,
    )

    validation_dataset = ShakespeareDataset(
        token_ids=validation_tokens,
        context_length=128,
        stride=128,
    )

    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, 64, shuffle=False)

    model_config = ModelConfig(
        vocabulary_size,
        embedding_dim=384,
        attention_dim=64,
        num_heads=6,
        hidden_dim=768,
        num_layers=5,
        dropout=0.2
    )
    model = GPT(model_config, use_dropout=True)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=0.1,
        lr=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    criterion = nn.CrossEntropyLoss()

    history_dict = train(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=25,
        validation_loader=validation_loader,
        scheduler=scheduler,
    )

    save_learning_curves(history_dict, LEARNING_CURVE_PATH)
    model.save(MODEL_PATH)

    
if __name__ == "__main__":
    main()

"""
Results from terminal:

high_capacity_model.py 
Initializing training:
Currently at epoch 1...
Training loss: 2.4974 | Validation loss: 2.2025 | LR: 0.001000
Currently at epoch 2...
Training loss: 2.0266 | Validation loss: 1.9080 | LR: 0.001000
Currently at epoch 3...
Training loss: 1.7905 | Validation loss: 1.7733 | LR: 0.001000
Currently at epoch 4...
Training loss: 1.6636 | Validation loss: 1.6930 | LR: 0.001000
Currently at epoch 5...
Training loss: 1.5835 | Validation loss: 1.6503 | LR: 0.001000
Currently at epoch 6...
Training loss: 1.5255 | Validation loss: 1.6003 | LR: 0.001000
Currently at epoch 7...
Training loss: 1.4812 | Validation loss: 1.5780 | LR: 0.001000
Currently at epoch 8...
Training loss: 1.4436 | Validation loss: 1.5640 | LR: 0.001000
Currently at epoch 9...
Training loss: 1.4135 | Validation loss: 1.5529 | LR: 0.001000
Currently at epoch 10...
Training loss: 1.3863 | Validation loss: 1.5420 | LR: 0.001000
Currently at epoch 11...
Training loss: 1.3628 | Validation loss: 1.5324 | LR: 0.001000
Currently at epoch 12...
Training loss: 1.3414 | Validation loss: 1.5297 | LR: 0.001000
Currently at epoch 13...
Training loss: 1.3211 | Validation loss: 1.5259 | LR: 0.001000
Currently at epoch 14...
Training loss: 1.3033 | Validation loss: 1.5242 | LR: 0.001000
Currently at epoch 15...
Training loss: 1.2864 | Validation loss: 1.5287 | LR: 0.001000
Currently at epoch 16...
Training loss: 1.2697 | Validation loss: 1.5219 | LR: 0.001000
Currently at epoch 17...
Training loss: 1.2530 | Validation loss: 1.5313 | LR: 0.001000
Currently at epoch 18...
Training loss: 1.2390 | Validation loss: 1.5332 | LR: 0.001000
Currently at epoch 19...
Training loss: 1.2238 | Validation loss: 1.5514 | LR: 0.000500
Currently at epoch 20...
Training loss: 1.1572 | Validation loss: 1.5559 | LR: 0.000500
Currently at epoch 21...
Training loss: 1.1292 | Validation loss: 1.5741 | LR: 0.000500
Currently at epoch 22...
Training loss: 1.1135 | Validation loss: 1.5811 | LR: 0.000250
Currently at epoch 23...
Training loss: 1.0688 | Validation loss: 1.6117 | LR: 0.000250
Currently at epoch 24...
Training loss: 1.0507 | Validation loss: 1.6284 | LR: 0.000250
Currently at epoch 25...
Training loss: 1.0386 | Validation loss: 1.6409 | LR: 0.000125
Best epoch: 16 | Lowest validation loss: 1.5219


Model Capacity, Regularization, and Generalization Report
=========================================================

The Tiny Shakespeare dataset contains approximately one million characters,
converted into character-level tokens. Several larger models have been
trained, but the previously trained smaller, regularized model has
consistently achieved better validation performance.

The central hypothesis is that larger models may have more capacity than
the available training data can effectively support.

1. Smaller Regularized Model previously trained in: regularized_model.py
----------------------------

The smaller model was trained for 20 epochs with a context length of 128,
a batch size of 64, dropout of 0.1, and AdamW weight decay of 0.1. A
training stride of 64 provided overlapping training windows.

Results:
- Final training loss: 1.2858
- Best validation loss: 1.4736
- Best epoch: 16
- Validation loss at epoch 20: 1.4752

After reaching its best validation loss at epoch 16, validation performance
remained relatively stable, suggesting that the model generalized reasonably
well to unseen data.

2. Larger High-Capacity Model
-----------------------------

The larger model had approximately 10 million parameters and was trained
for 25 epochs with a context length of 128 and a batch size of 64. It used
dropout of 0.2, AdamW weight decay of 0.1, an initial learning rate of
0.001, and a ReduceLROnPlateau scheduler. Its training stride was 128.

Results:
- Final training loss: 1.0386
- Best validation loss: 1.5219
- Best epoch: 16
- Validation loss at epoch 25: 1.6409

Although the larger model achieved a substantially lower training loss,
its validation loss increased after epoch 16. Repeated learning-rate
reductions did not prevent this deterioration.

3. Comparison and Interpretation
---------------------------------

The smaller model achieved a better best validation loss of 1.4736,
compared with 1.5219 for the larger model. This is consistent with
overfitting in the larger model: it continued fitting the training data
more closely without improving its performance on unseen data.

Interestingly, both models achieved their lowest validation loss at epoch
16, despite their different capacities and training configurations.
However, the smaller model's validation loss remained relatively stable,
while the larger model's increased substantially.

The fact that several larger models have also underperformed the smaller
model strengthens the hypothesis that additional model capacity is not
beneficial for this dataset under the configurations tested.

4. Conclusion
----------------------------

So far, the smaller regularized model has provided better validation
performance than all of the the larger models tested, while requiring less
computational capacity. The results suggest that a larger model can fit
the training corpus more closely without learning patterns that generalize
better to unseen text.

"""