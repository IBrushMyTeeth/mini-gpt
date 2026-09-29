from model.gpt import GPT
from pathlib import Path
from tokenization.tokenizer import Tokenizer

from generation.prompts import SHAKESPEARE_PROMPTS
from generation.utils import generate_completions


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "regularized.pt"
TOKENIZER_PATH = ROOT / "data" / "shakespeare_char_tokenizer_config.pt"

NEW_TOKENS = 200

def main():
    model = GPT.load(MODEL_PATH)
    tokenizer = Tokenizer.load(TOKENIZER_PATH)

    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=SHAKESPEARE_PROMPTS,
        new_tokens=NEW_TOKENS
    )

    for i, (prompt, completion) in enumerate(
            zip(SHAKESPEARE_PROMPTS, completions), start=1
    ):
        print("=" * 60)
        print(f"Prompt {i}")
        print("=" * 60)
        print(prompt)
        print()
        print("Completion:")
        print(completion)
        print()

if __name__ == "__main__":
    main()

"""
Results from terminal:

============================================================
Prompt 1
============================================================
KING HENRY:
What news from the north, good my lord?

Completion:
KING HENRY:
What news from the north, good my lord?

KING EDWARD IV:
And so much many more than the common of the common of the state,
And the seal'd of the season of the season
And shall be the stroke of the season of the seas,
And the season of the 

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Completion:
Now is the winter of our discontent.

KING EDWARD IV:
Why, then the common of the common soundly of the state
And the state of the season of the season,
And the seal'd of the season of the season
And shall be the stroke of the season o

============================================================
Prompt 3
============================================================
The night is dark

Completion:
The night is darkeneral of the state of the state,
And so much a soldiers of the sun and souls of the state
To see the sea-sick of the season of the seas,
And the season of the season of the season
And shall be the st

============================================================
Prompt 4
============================================================
To be, or not to be, that is the question:

Completion:
To be, or not to be, that is the question:
I will not so man that the state of the state
That shall be so fall of the state of the season
And shall be the stroke of the season of the seas,
And the season of the season of the season
And shall 

============================================================
Prompt 5
============================================================
Fie, fie, upon this quiet life! I want work.

Completion:
Fie, fie, upon this quiet life! I want work.

PERDITA:
I was the common of the common that the stroke
To see the world that the world shall be so fall.

DUKE VINCENTIO:
I will not so, I say the senators of the season
And save the souls of the s

============================================================
Prompt 6
============================================================
FIRST CITIZEN:
Before we proceed any further, hear me speak.

Completion:
FIRST CITIZEN:
Before we proceed any further, hear me speak.

QUEEN MARGARET:
I will not stay the strange of the state of the season
And shall be so fall of the season of the season
And so souls and the state of the seasons of the season
And shall be the strok

============================================================
Prompt 7
============================================================
The king shall

Completion:
The king shall be so much a soldier to the state
And shall be the stroke of the state of the state
And that the state of the state of the season
And shall be the stroke of the season of the seas,
And the season of 

============================================================
Prompt 8
============================================================
Romeo:

Completion:
Romeo:
I say the common of the state of the state,
And so much a soldiers of the sun as the souls,
And the stroke of the season of the season
And shall be the stroke of the season of the seas,
And the seaso


Generation comparison between the baseline and regularized models:

Even though adding regularization allowed the model to generalize better to
the validation data and substantially reduced the divergence between training
and validation loss, the effect on greedy generation is different. With
argmax decoding, the regularized model produces considerably more repetitive
text. Phrases such as "the state of the season", "the season of the season",
and "the stroke of the season" are repeatedly generated across several
prompts.

This repetition may indicate that the model has learned strong reusable
character-level patterns from the Shakespeare corpus. The training data
contains many recurring structures involving sequences such as "the", "of
the", "and", and "shall". Under greedy decoding, always selecting the most
probable next character can cause the model to repeatedly follow these
high-probability local patterns. The repetition therefore does not
necessarily mean that the model has forgotten earlier text in its context;
rather, it may be repeatedly following learned character-level templates
that remain highly probable given the current context.

For example, Shakespeare contains enormous numbers of patterns involving:
the ...
of the ...
and ...
shall ...
the ... of the ...

In comparison, the baseline model produces more varied dialogue, including
character names, dialogue structures, and Shakespearean-style vocabulary.
However, the baseline checkpoint was saved at epoch 20, after substantial
overfitting had already occurred. Its lower training loss and stronger
specialization to the training corpus can make its generated text appear
more convincing and varied, since it has learned more of the corpus-specific
patterns and structures.

Therefore, the seemingly better generation from the baseline does not
contradict the improved validation performance of the regularized model.
Rather, the experiment demonstrates a difference between generalization
performance and behaviour under a specific decoding strategy. The
regularized model generalizes better to unseen validation data, but under
deterministic argmax decoding it is more prone to repetitive sequences.

Further generation experiments using temperature and sampling are therefore
useful for determining whether this repetitive behaviour is primarily a
consequence of the greedy decoding strategy.
"""