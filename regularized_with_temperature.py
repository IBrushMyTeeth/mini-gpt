from pathlib import Path

import torch

from generation.utils import generate_completions_with_temp
from model.gpt import GPT
from tokenization.tokenizer import Tokenizer


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "regularized.pt"
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

KING EDWARD IV:
Nay, my lord, I cannot be to heard and my back,
And curse the cause of All-sheet to be done
And better to find me to his dead, I make the man's will.

ROMEO:
Peter, there to make the

Temperature: 1.0
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING EDWARD IV:
Nay, it commend to-night! will I must be old Gaunt,
Such as these contempt my state from you made the dear of forsake;
Now, not upon buy love's granting how most: is your conscient?


Temperature: 1.5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING EDWARD IV:
Day fittler else, whose this, having,cannot his my ccusion;
Gracuo1 great: The tyrannor of yourselves?

KING LEUMCILIO:
Tush! bluspasses we o'erward with at my trues!
LifCymUS:
Wora 

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Temperature: 0.5
------------------------------------------------------------
Now is the winter of our discontent.

KING EDWARD IV:
So sir, I were now now to speak my heart back,
And curse the cause of my purpose.

KING RICHARD III:
So save for the sun stands of the glory shame
Which are they are not to make a m

Temperature: 1.0
------------------------------------------------------------
Now is the winter of our discontent abuse.

Third Servant:
Becomeet you not by thy hand; 'cannot his my cousin,

GLOUCESTER:
I put thy garland, I must tell thee, but stay as too!

QUEEN MARGARET:
No pray, but love Romeo Lamaster!

DUKE

Temperature: 1.5
------------------------------------------------------------
Now is the winter of our discontent about.
Insibunding, Elittler,ely--

GLOUCESTER:
I mustcan.
Was fury ccusione
He success Abravaged come too appulture.

GREGORY:
Uts, you susper I moan tell howshire awake:
Groom Romeo: bawd!

Second 

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
I cannot send to the banish'd the way;
What shall come to come on the bearing of the deed
And soul sound the mark of souls and to him him o'erchange
And the stroke of many more of 

Temperature: 1.0
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING LEWIS XI:
It, he that mean thou lady to Buckingham?

JULIET:
I curt not save the mind upon you from you,
Thus is death, but stay as to dram to hole.

PRINCE EDWARD:
I love so, pleasant, have a 

Temperature: 1.5
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING LET:
Fair, whence thou else, wholest toy's far hcam.
Was fury ccusions, shame I thinKnog east;
What you made tell to forbum Campinss.
Cork, moathee, mustsh gone.
I have persons you wake't, paud


Generation with temperature sampling:

Temperature sampling provides a different picture of the regularized model's
generation behaviour than greedy decoding. When sampling is used, the strong
repetitive patterns observed with argmax decoding largely disappear, showing
that the model can produce more diverse text when it is not restricted to the
single most probable character.

At a temperature of 0.5, the generations remain conservative and retain
recognizable Shakespearean dialogue structures. At 1.0, the model produces
greater variation in character names, vocabulary, and dialogue while still
maintaining recognizable Shakespearean characteristics. At 1.5, the output
becomes substantially more random, with malformed words and increasingly
incoherent sequences.

Although the generated text is not consistently grammatically or semantically
correct, the results are quite good considering the model's small size,
character-level tokenizer, limited dataset, and computational constraints.
Among the models trained in this project, this smaller regularized model also
achieved the best validation performance and produced some of the more
coherent generated samples.

Overall, a moderate temperature, particularly around 1.0, provides a good
balance between repetition and randomness. The experiment demonstrates that
the model learned recognizable Shakespearean vocabulary, dialogue structures,
character names, and stylistic patterns, while achieving the best overall
generalization of the models tested.
"""