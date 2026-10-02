from model.gpt import GPT
from pathlib import Path
from tokenization.tokenizer import Tokenizer

from generation.prompts import SHAKESPEARE_PROMPTS
from generation.utils import generate_completions


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "regularized_ctx256.pt"
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
Results from the terminal:

============================================================
Prompt 1
============================================================
KING HENRY:
What news from the north, good my lord?

Completion:
KING HENRY:
What news from the north, good my lord?

KING RICHARD IIIII:
The state of the king of the strength of the state,
And shall be the seal'd of my son,
And they shall be a man and the state,
That thou wilt be the country of the state,
The stat

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Completion:
Now is the winter of our discontented with them
To stand the state of the season of the sea
Which he will be so soon of the counterfeit of the
secret of the state of the straight of the
seath of the state of the season of the people,
T

============================================================
Prompt 3
============================================================
The night is dark

Completion:
The night is dark'd of the country's heart
To see the state of the state of the strong.

KING RICHARD IIII:
What shall be the straight of the seal's death,
And the strange of the seath of the state,
That thou wilt be 

============================================================
Prompt 4
============================================================
To be, or not to be, that is the question:

Completion:
To be, or not to be, that is the question:
The straight of the country's son,
And the matter of the court of the state,
The state of the seat of the state,
That we were a straight of the prince
That he was a man and stand of the state
To see 

============================================================
Prompt 5
============================================================
Fie, fie, upon this quiet life! I want work.

Completion:
Fie, fie, upon this quiet life! I want work.

KING RICHARD IIII:
Then I will not be so.

KING RICHARD IIII:
What shall I shall be the state of the sea
To such a stronge of the state of the state.

KING HENRY VI:
The consul of the seath of the s

============================================================
Prompt 6
============================================================
FIRST CITIZEN:
Before we proceed any further, hear me speak.

Completion:
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD IIII:
What shall I do not be so much as the state
To see the sense of the common of my son.

KING RICHARD III:
The state of the state of the state,
And the state of the state of the sea

============================================================
Prompt 7
============================================================
The king shall

Completion:
The king shall be so seen to him and the state
To see him the state of the state of the sea
To see him the people of the state of the seat,
The sent of the state of the world,
And the state of the state of the secr

============================================================
Prompt 8
============================================================
Romeo:

Completion:
Romeo: I will say 'I will be so.

PETRUCHIO:
The way will seek thee the world in the state
And the state of the world, and the state,
That we shall be so stand and the seath of the seast.

PAULINA:
I would



Generation results for the model trained with context length 256 using
greedy decoding:

The previously observed issue with recurring repetitions across different
prompts with greedy decoding has greatly improved but still exists.Phrases
such as "the state of the state", "the state of the sea", and "the straight
of the state". This is however much better than what we see with the previous
context length of 128 where we had sequences such as:

"And the seal'd of the season of the season
And shall be the stroke of the season of the seas,
And the season of the "

or

"Why, then the common of the common soundly of the state
And the state of the season of the season,
And the seal'd of the season of the season
And shall be the stroke of the season o"

These improved results suggest that the model has learned local character-level
and textual patterns but still struggles to combine them into meaningful,
sustained passages.

Overall, the model demonstrates recognizable features of Shakespearean text,
but its generations remain repetitive and often incoherent.
"""