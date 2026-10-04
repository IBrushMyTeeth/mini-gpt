from pathlib import Path

import torch

from generation.utils import generate_completions_with_temp
from model.gpt import GPT
from tokenization.tokenizer import Tokenizer


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "high_capacity.pt"
TOKENIZER_PATH = ROOT / "data" / "shakespeare_char_tokenizer_config.pt"

NEW_TOKENS = 200
TEMPERATURES = [0.5, 1.0, 1.5]

PROMPTS = [
    "KING HENRY:\nWhat news from the north, good my lord?",
    "Now is the winter of our discontent",
    "FIRST CITIZEN:\nBefore we proceed any further, hear me speak.",
]

def main():

    model = GPT.load(MODEL_PATH)
    tokenizer = Tokenizer.load(TOKENIZER_PATH)

    for i, prompt in enumerate(PROMPTS, start=1):
        print("=" * 60)
        print(f"Prompt {i}")
        print("=" * 60)
        print(prompt)
        print()

        for temperature in TEMPERATURES:
            torch.manual_seed(42)

            completion = generate_completions_with_temp(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                new_tokens=NEW_TOKENS,
                temperature=temperature,
            )[0]

            print(f"Temperature: {temperature}")
            print("-" * 60)
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

Temperature: 0.5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD III:
I see to me the world I will not so.

GLOUCESTER:
I have provost to my grace to the common of the head,
And for the sustic of those that haste
To sent to see the strong to my son: 

Temperature: 1.0
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD III:
I could reely think thy wife, I do, and back,
As I news, then more 'gainst to scorn too,
Let thee had patient that hast stir humour tender
Where now I have person there; or have be

Temperature: 1.5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD III:
I could, men
Nurse lady to Bushy, havingback, my lord; thou lay 1eguL
Were tears a From York, upon what forboots:
Harks I not, more ten might give mine: thou thousandst torms, or--

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Temperature: 0.5
------------------------------------------------------------
Now is the winter of our discontent
To far his brother, and thou encounters' tears,
And shall be the more princely than the prepared of the deep
And she had safellow'd him so like a moletter
Than he will not the strong of a more suppos

Temperature: 1.0
------------------------------------------------------------
Now is the winter of our discontent
Of faction? and I'll fit you encountent.

ROMEO:
I do, and be sender curferal, not the more.

MENENIUS:
Ao, I must tell to forblood: you shall be to honest honour
In make a love some less; or hanging

Temperature: 1.5
------------------------------------------------------------
Now is the winter of our discontent are:
Then i' this very thus men
That's all time, firstcom. Was't by ccurish;
Grasmo1s this: These conful oyal maniBellad?

LUCIO:
Came a sudder I do hote.

CLAUDIO:
What hath persets your content in 

============================================================
Prompt 3
============================================================
FIRST CITIZEN:
Before we proceed any further, hear me speak.

Temperature: 0.5
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
I cannot tell thee the time to my heart;
And yet I shall be say 'I' the content.'

FLORIZEL:
And I will for my son.

CLARENCE:
O, she shall shall be the way to-morrow.

CAMILLO:
My

Temperature: 1.0
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
I can commend to-night to you first and back,
And cures to slaugh him be sheep to from you.

ESBRUTES:
If he make you shall be to home.

CORIOLANUS:
I have person to be too hanging

Temperature: 1.5
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
I come or else, wholes this harm; hear old George! Hall them succes my
KING LEWIS XF RD II:
Thus, Marcius, builia, hast line I do hote.

CLAUDIO:
What hath persets your contempt?--


High-Capacity Model Generation Report
======================================

At temperature 0.5, the model produces relatively conservative output.
The generated text contains recognizable Shakespearean vocabulary,
character names, dialogue formatting, and sentence structures. However,
many sentences are only locally plausible and do not remain semantically
coherent.

At temperature 1.0, the model produces greater variation while still
maintaining recognizable Shakespearean style. Character names, dialogue,
punctuation, and vocabulary often look convincing, although malformed words
and grammatically incorrect sentences remain common. This temperature
provides the best balance between diversity and control in these experiments.

At temperature 1.5, the output becomes substantially more random. The model
produces malformed words, unusual character names, corrupted character
sequences, and increasingly fragmented sentences. This demonstrates the
expected trade-off between diversity and coherence when temperature is
increased.

Overall, the high-capacity model has clearly learned important statistical
patterns from the Shakespeare corpus, including vocabulary, punctuation,
dialogue structure, and character names. However, it does not consistently
produce coherent longer passages.

The generation results are consistent with the validation experiments. The
high-capacity model achieved a much lower training loss of 1.0386 but a worse
best validation loss of 1.5219, compared with 1.4736 for the smaller
regularized model.

Therefore, the additional model capacity appears to improve the model's
ability to fit the training data without providing a corresponding
improvement in generalization or generation quality.
"""