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

        print(f"Temperature: {TEMPERATURE}")
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

Temperature: 0.5
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

Temperature: 0.5
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

Temperature: 0.5
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



Context-length ablation using a training passage to test memorization:

A passage taken directly from the training data was used to investigate how
the 256-context model behaves when given 64, 128, and 256 characters of
preceding context. The same prompts were evaluated using both temperature
sampling and greedy decoding.

With temperature sampling at 0.5, increasing the prompt length allowed the
model to reproduce progressively more of the original training passage before
generating new text. The 256-character prompt reproduced substantially more
of the passage than the shorter prompts. Since the passage is part of the
training data, this reproduction demonstrates memorization rather than
generalization.

After leaving the memorized passage, the sampled generations maintained
recognizable Shakespearean dialogue structures, including speaker names,
punctuation, and Shakespearean-style vocabulary. The model therefore retains
a degree of structural consistency even after it begins generating text not
directly copied from the supplied passage.

The greedy decoding results show a different behaviour. Although the model
again reproduces progressively more of the training passage as the context
length increases, the generated continuation quickly falls into highly
repetitive character-level patterns. With the 256-character prompt, the
model repeatedly generates sequences such as "the senate of the seath of
the seath", demonstrating that the longer context does not prevent greedy
decoding from becoming trapped in a high-probability repetition loop.

The contrast between the two decoding strategies is therefore particularly
informative. The model appears to have a broader distribution of plausible
continuations than greedy decoding exposes. Temperature sampling allows the
model to move between different probable character sequences, while argmax
decoding repeatedly selects the same locally probable patterns.

Overall, the experiment demonstrates that the 256-context model can memorize
and reproduce substantially longer sequences from its training data, but the
additional context alone does not resolve the repetitive behaviour observed
under greedy decoding. Because the tested passage belongs to the training
data, a corresponding experiment on unseen validation text would be required
to determine whether the longer context provides a genuine generalization
benefit.
"""