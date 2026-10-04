from pathlib import Path

import torch

from generation.utils import generate_completions_with_temp, generate_completions
from model.gpt import GPT
from tokenization.tokenizer import Tokenizer


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "regularized_ctx256.pt"
TOKENIZER_PATH = ROOT / "data" / "shakespeare_char_tokenizer_config.pt"

NEW_TOKENS = 200
TEMPERATURE = 0.5

# This passage is from the training set
PASSAGE = """Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth him; he cannot swear, but it cheques him;
he cannot lie with his neighbour's wife, but it
detects him: 'tis a blushing shamefast spirit that
mutinies in a man's bosom; it fills one full of
obstacles: it made me once restore a purse of gold
that I found; it beggars any man that keeps it: it
is turned out of all towns and cities for a
dangerous thing; and every man that means to live
well endeavours to trust to himself and to live
without it.
"""

PROMPTS = [
    PASSAGE[:64],
    PASSAGE[:128],
    PASSAGE[:256],
]

def main():

    model = GPT.load(MODEL_PATH)
    tokenizer = Tokenizer.load(TOKENIZER_PATH)

    for i, prompt in enumerate(PROMPTS, start=1):
        print("=" * 60)
        print(f"Prompt {i} ({len(prompt)} characters)")
        print("=" * 60)
        print(prompt)
        print()

        torch.manual_seed(42)

        completion = generate_completions_with_temp(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            new_tokens=NEW_TOKENS,
            temperature=TEMPERATURE,
        )[0]

        print(f"Temperature: {TEMPERATURE}")
        print("-" * 60)
        print(completion)
        print()

    for i, prompt in enumerate(PROMPTS, start=1):
        print("=" * 60)
        print(f"Prompt {i} ({len(prompt)} characters)")
        print("=" * 60)
        print(prompt)
        print()

        torch.manual_seed(42)

        completion = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=[prompt],
            new_tokens=NEW_TOKENS,
        )[0]

        print("-" * 60)
        print(completion)
        print()


if __name__ == "__main__":
    main()


"""
============================================================
Prompt 1 (64 characters)
============================================================
Second Murderer:
I'll not meddle with it: it is a dangerous thin

Temperature: 0.5
------------------------------------------------------------
Second Murderer:
I'll not meddle with it: it is a dangerous thing
Before me be dishonour'd or else of the world.

KING RICHARD IIII:
O my lord, I more than I have thank your duke
Being to me burther.

BUCKINGHAM:
O blest honour prayers and your soul!

CAPULET:
Wel

============================================================
Prompt 2 (128 characters)
============================================================
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth 

Temperature: 0.5
------------------------------------------------------------
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth a more than we did not we come on.

Provost:
Not a man common of my brother, they mother.

KING RICHARD III:
Not such as the charge of his dead?

BUCKINGHAM:
Well, my lord: they they will well have a 

============================================================
Prompt 3 (256 characters)
============================================================
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth him; he cannot swear, but it cheques him;
he cannot lie with his neighbour's wife, but it
detects him: 'tis a blushing shamefast

Temperature: 0.5
------------------------------------------------------------
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth him; he cannot swear, but it cheques him;
he cannot lie with his neighbour's wife, but it
detects him: 'tis a blushing shamefast as face,
And the was a true of convey of this is all,
And in the manning of the sun of the purpose.

LADY ANNE:
Angelo, that he may be a sudden and as the man
As is a servant.

PRINCE EDWARD:
So pard


============================================================
Prompt 1 (64 characters)
============================================================
Second Murderer:
I'll not meddle with it: it is a dangerous thin

------------------------------------------------------------
Second Murderer:
I'll not meddle with it: it is a dangerous thing
The streets of the strong of the seat of the state,
That we will our state of the state,
The state of the state of the seat of the state
Of the secret of the senators of the court,
And the state of 

============================================================
Prompt 2 (128 characters)
============================================================
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth 

------------------------------------------------------------
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth a man all the seal of the state,
and the world of the seast of the commonstrong.

CAMILLO:
The state of the sea of the people,
And the state of the seath of the seath,
The senate of the seath of the s

============================================================
Prompt 3 (256 characters)
============================================================
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth him; he cannot swear, but it cheques him;
he cannot lie with his neighbour's wife, but it
detects him: 'tis a blushing shamefast

------------------------------------------------------------
Second Murderer:
I'll not meddle with it: it is a dangerous thing: it
makes a man a coward: a man cannot steal, but it
accuseth him; he cannot swear, but it cheques him;
he cannot lie with his neighbour's wife, but it
detects him: 'tis a blushing shamefaster, and the seath
and the senate of the seath of the seath,
and the senate of the seath of the seath,
and the senate of the seath of the seath,
and the senate of the seath of the seath,
and the senate


Training-Data Memorization and Decoding Experiment
===================================================

A passage taken directly from the training data was used to examine how the
256-context model behaves with 64, 128, and 256 characters of preceding
context. Both temperature sampling and greedy decoding were tested.

With temperature sampling at 0.5, longer prompts allow the model to continue
the supplied passage for somewhat longer before generating new text. However,
the model does not reproduce large portions of the passage exactly. This
suggests some ability to memorize training sequences, but not complete
memorization of the passage.

Once the model moves beyond the supplied text, the generations contain
recognizable Shakespearean vocabulary, dialogue structure, and punctuation.
This indicates that the model has learned broader patterns from the dataset
rather than simply memorizing the tested passage.

Greedy decoding produces a different limitation. The model quickly falls into
repetitive character-level patterns, with the 256-character prompt eventually
producing sequences such as "the senate of the seath of the seath". This shows
how repeatedly selecting the most probable next character can lead to
repetition loops.

Overall, the experiment suggests limited training-data memorization rather
than direct reproduction of the entire passage. Temperature sampling produces
much more varied continuations than greedy decoding, while the longer context
allows the model to make use of more preceding information. However, this
experiment cannot establish whether a longer context improves generalization,
because the tested passage was part of the training data. An equivalent
experiment using unseen validation text would be needed to evaluate that.
"""