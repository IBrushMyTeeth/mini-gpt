# Character-Level Shakespeare Transformer
 
A small GPT-style, decoder-only transformer trained from scratch on the Tiny Shakespeare dataset, at the character level.
 
This is an educational project. Instead of relying on a high-level transformer library, the tokenizer, model architecture, training loop, checkpointing, evaluation, and text generation were all implemented and inspected step by step. The goal was not a state-of-the-art model, but a practical understanding of how a transformer language model works, how training choices affect generalization, and what happens when a model is larger than its dataset can support.
 
**Best validation loss: 1.4736** (small regularized model, ~20 epochs, CPU only).
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset and Tokenization](#dataset-and-tokenization)
- [Training Pipeline](#training-pipeline)
- [Experiments](#experiments)
- [Text Generation](#text-generation)
- [Key Lessons](#key-lessons)
- [Computational Constraints](#computational-constraints)
- [Why Character-Level Modelling?](#why-character-level-modelling)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
---
 
## Overview
 
The project implements the full pipeline for a character-level language model:
 
```
data → tokenizer → tensors → transformer → training → validation
     → checkpointing → serialization → generation → evaluation
```
 
Concretely, it covers:
 
1. Downloading the Tiny Shakespeare dataset
2. Building a character-level tokenizer
3. Converting the text to integer tokens and saving them as PyTorch tensors
4. Defining a GPT-style decoder-only transformer
5. Training and validating the model
6. Serializing trained models
7. Loading models and generating text
8. Evaluating models quantitatively (loss) and qualitatively (samples)
9. Experimenting with model capacity, regularization, optimization, learning-rate scheduling, and decoding strategies
The work followed an iterative loop:
 
```
Implement → Train → Measure → Inspect → Find a problem
         → Form a hypothesis → Change one thing → Train again
```
 
The project was as much about understanding model behaviour as about reaching a good validation loss.
 
---
 
## Model Architecture
 
The model is a decoder-only transformer for autoregressive character-level language modelling. The architecture is assembled explicitly from its components. PyTorch building blocks such as `nn.Embedding`, `nn.Linear`, dropout, and layer normalization are used where appropriate.
 
**Components**
 
- Token embeddings
- Positional embeddings
- Causal self-attention (first as single heads, then combined into multi-head attention)
- Layer normalization
- Feed-forward MLP sub-blocks
- Residual connections
- Dropout
- Linear projection to vocabulary logits
**Structure**
 
```
Input characters
      │
      ▼
Character tokenizer
      │
      ▼
Token embeddings + positional embeddings
      │
      ▼
┌───────────────────────────┐
│ Transformer block  (×N)   │
│                           │
│ LayerNorm                 │
│    ↓                      │
│ Multi-head self-attention │
│    ↓                      │
│ Residual connection       │
│    ↓                      │
│ LayerNorm                 │
│    ↓                      │
│ MLP                       │
│    ↓                      │
│ Residual connection       │
└───────────────────────────┘
      │
      ▼
Linear vocabulary projection
      │
      ▼
Next-character probabilities
```
 
---
 
## Dataset and Tokenization
 
The project uses the Tiny Shakespeare dataset, which contains roughly one million characters.
 
Each unique character is assigned an integer ID, and the resulting integer sequence is stored as PyTorch tensors, so the raw text file does not need to be re-processed during training.
 
```
Tiny Shakespeare → character tokenizer → integer sequence
                 → PyTorch tensor → train / validation splits
```
 
A character-level tokenizer keeps the pipeline simple and transparent, which makes it easy to see exactly what the transformer is learning. The trade-off is that the model must learn spelling, word boundaries, and morphology one character at a time. Subword tokenization would be more efficient, but also more computationally demanding (see [Future Work](#future-work)).
 
---
 
## Training Pipeline
 
The pipeline supports:
 
- Train/validation splitting
- Context-window sampling with a configurable stride (overlapping or non-overlapping windows)
- Configurable batch size and context length
- AdamW optimization with weight decay
- Dropout
- Learning-rate scheduling
- Validation after every epoch
- Checkpointing and serialization
- Text generation from saved models
An early lesson shaped the pipeline: **the model from the last epoch is not necessarily the best one.** Validation loss can improve and then deteriorate as training continues, so the pipeline keeps the checkpoint with the best validation loss rather than the final state.
 
---
 
## Experiments
 
### 1. Model capacity
 
The experiment compared a smaller regularized model against larger models, including one with roughly 10 million parameters.
 
| Setting | Smaller model | Larger model |
| --- | --- | --- |
| Context length | 128 | 128 |
| Batch size | 64 | 64 |
| Dropout | 0.1 | 0.2 |
| AdamW weight decay | 0.1 | 0.1 |
| Initial learning rate | — | 1e-3 |
| LR scheduler | — | ReduceLROnPlateau |
| Training stride | 64 | 128 |
| Epochs | 20 | 25 |
 
| Metric | Smaller model | Larger model |
| --- | --- | --- |
| Final training loss | 1.2858 | 1.0386 |
| **Best validation loss** | **1.4736** | 1.5219 |
| Best epoch | 16 | 16 |
| Validation loss at last epoch | 1.4752 (epoch 20) | 1.6409 (epoch 25) |

The larger model always fit the training data far better but generalized worse. Multiple large configurations were tried, but the smaller regularized model consistently won on validation loss.
 
When comparing the two saved models from training scripts regularized_model.py and
high_capacity_model.py:
Both models reached their lowest validation loss at around the same point in training (epoch 16), but behaved very differently afterwards. The smaller model stayed stable, while the larger model kept driving training loss down as its validation loss climbed. By epoch 25 the gap between training and validation loss was about 0.6.
 
For context, the larger model has on the order of 10 parameters per training token, which is far more capacity than a ~1M-character corpus can justify. These results are consistent with the idea that this dataset does not benefit from much more capacity under the configurations tested. This is an empirical finding for these settings, not a universal rule.
 
### 2. Regularization
 
Dropout and AdamW weight decay were explored as ways to control overfitting. The larger models showed that extra capacity makes it easy to lower training loss without improving validation performance.
 
More regularization is not automatically better, either. Capacity, dropout, weight decay, learning rate, training duration, and dataset size all interact. The useful outcome was not a magic value, but learning to read the training and validation curves and adjust accordingly.
 
### 3. Learning-rate scheduling
 
A `ReduceLROnPlateau` scheduler (factor 0.5, patience 2) lowered the learning rate when validation loss stopped improving. The idea was to take larger steps early and smaller steps late.
 
In practice it did not prevent the larger model from overfitting. With a patience of 2, the scheduler also reacted after validation loss had already passed its minimum. This reinforced a general point:
 
> Better optimization can improve how well a model fits the training data without improving generalization.
 
---
 
## Text Generation
 
Trained models were loaded from their serialized checkpoints and evaluated qualitatively by generating Shakespeare-style text. Samples revealed behaviours that validation loss alone cannot show, and decoding strategy turned out to matter a great deal.
 
### Greedy (argmax) decoding
 
Always choosing the most probable character produced strong repetition and loops. The model found highly probable local sequences and kept following them. This initially suggested a lack of diversity in the model, but the sampling experiments below showed that was not the full story.
 
### Top-k sampling
 
Top-k sampling restricts the next-character distribution to the *k* most probable candidates and samples among them.
 
- **Small *k* (e.g. 1):** tightly constrained output that kept recognizable Shakespearean structure: character names, dialogue, and vocabulary.
- **Larger *k* (e.g. 10–20):** more variation, but also unusual character names, fragmented sentences, and sequences that did not form meaningful phrases.
This illustrates the trade-off between staying on high-probability continuations and exploring more of the learned distribution.
 
### Temperature sampling
 
Sampling from the full distribution largely removed the loops seen with greedy decoding, which indicates that the repetition came from always picking the single most likely character, not from an inability to produce variety.
 
| Temperature | Observed behaviour |
| --- | --- |
| 0.5 | Conservative; recognizable Shakespearean structure |
| 1.0 | More variation while retaining reasonable structure |
| 1.5 | Much more randomness: malformed words, odd names, incoherent sequences |
 
A moderate temperature gave the best balance between deterministic repetition and excessive randomness.
 
> Generation quality depends not only on the model, but also on how its probability distribution is decoded.
 
---
 
## Key Lessons
 
- **Bigger is not automatically better.** Larger models reached lower training loss but worse validation loss. Capacity has to be considered relative to the amount of data.
- **More parameters can mean more memorization.** Extra capacity let the model fit the training distribution more closely without learning anything more general.
- **The final epoch is not necessarily the best model.** Always checkpoint on the best validation loss.
- **Training curves are diagnostic tools.** Comparing training and validation loss shows whether a change is actually helping. Targeted, one-at-a-time changes beat blind tweaking.
- **Regularization requires balance.** Dropout and weight decay help, but their effect depends on model size, data, optimization settings, and training length.
- **Schedulers refine training but do not fix overfitting.** A learning-rate schedule cannot solve a fundamental generalization problem.
- **Decoding matters.** Greedy decoding hid much of what the model had learned; temperature and top-k sampling revealed far more variety.
- **Quantitative and qualitative evaluation complement each other.** Validation loss measures generalization; samples reveal repetition, malformed words, and style.
---
 
## Computational Constraints
 
The project was developed on limited, CPU-based hardware, which constrained:
 
- Model size
- Training duration
- The number of experiments
- Context length
- The scale of the dataset
- The use of computationally expensive tokenization schemes
Despite this, the best model reached a validation loss of about **1.47**. This is in the range reported by comparable educational Tiny Shakespeare implementations, though direct comparisons should be made cautiously, since architecture, tokenizer, context length, training procedure, data processing, and evaluation methodology all differ between projects.
 
---
 
## Why Character-Level Modelling?
 
The character tokenizer was a deliberate choice to keep the whole modelling process visible. The model has to learn everything from individual characters:
 
```
c → h → a → r → a → c → t → e → r
```
 
rather than from pre-built words or subword units. This makes the project a good vehicle for understanding tokenization, vocabulary construction, embeddings, autoregressive prediction, context windows, attention, sampling, and language-model probabilities.
 
The cost is efficiency. A subword tokenizer would represent language more compactly and could give better results, but would add complexity and compute requirements. For this project, simplicity was the point.
 
---
 
## Future Work
 
**Data and sampling**
 
- **Random window offsets.** Training currently uses fixed strides, so the model sees the same chunk boundaries every epoch. Sampling random start positions would act as free data augmentation and directly counter memorization.
- **More data.** Tiny Shakespeare is very small by modern standards. More text would add linguistic variety and could make additional capacity worthwhile.
**Model and optimization**
 
- **Systematic capacity ablations.** Compare model sizes (e.g. different embedding dimensions and layer counts) by best validation loss under otherwise identical settings.
- **Cosine learning-rate decay with warmup,** and a somewhat lower peak learning rate, as a more predictable alternative to `ReduceLROnPlateau`.
- **Automated early stopping** so that epochs after the validation minimum are not wasted on CPU time.
- **Longer context windows** to capture longer-range structure (at higher compute cost).
**Tokenization and infrastructure**
 
- **Subword tokenization,** probably the most significant modelling change available from this point.
- **GPU access,** enabling larger models, longer contexts, and wider hyperparameter searches.
- **Experiment tracking.** Automate hyperparameter sweeps and record metadata, checkpoints, and validation curves systematically.
---
 
## Conclusion
 
This project produced a complete, working character-level transformer pipeline built from scratch, from raw text to generated samples. Under CPU-only constraints it reached a best validation loss of approximately **1.4736** on Tiny Shakespeare, using a small regularized model that generalized better than every larger model tested.
 
More important than the number is what the experiments showed: a larger model does not automatically generalize better, lower training loss does not mean a better model, the best checkpoint may come before the final epoch, regularization needs balance, and decoding choices can change the generated text dramatically.
 
The main outcome is the process: building the system, testing it, observing its failures, and using those observations to improve it.