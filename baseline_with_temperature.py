from pathlib import Path

import torch

from generation.utils import generate_completions_with_temp
from model.gpt import GPT
from tokenization.tokenizer import Tokenizer


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "baseline.pt"
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
This experiment investigates how temperature affects text generation.
Unlike the previous experiment, where the most probable next character
was always selected, this experiment samples the next character from the
model's probability distribution.

Three temperatures are tested:

- 0.5: more conservative and predictable generation
- 1.0: standard sampling
- 1.5: more random and diverse generation

The same three prompts are used for each temperature, with 200 new tokens
generated per completion. The random seed is reset before each generation
to make the results reproducible.


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
By my consent, I hold to thee all thee.

QUEEN ELIZABETH:
These all the prince that to the more than honour'd.

LUCIO:
Good sir, let him home, be so sooth.

ISABELLA:
Be prince, be

Temperature: 1.0
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD III:
By my correction?

QUEEN MARGARDIUS:
This is my blood; not slay the more.

LADY ANNE:
Ten the issues hot her? that has he not stoad term home.

KING RICHARD II:
What hast thou pity

Temperature: 1.5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD III:
By my correct--

GLOUCESTER:
I must be.
Was ever thus; thou lame1ted Apoth,
Steing from his louth he diest but Clarence.
Why I do hate: must he goes
I have persons lost; or havoide

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Temperature: 0.5
------------------------------------------------------------
Now is the winter of our discontent to far,
That this that the room of wrongs and such
A lawful assure and his beast of her proper and his,
And in the trust proverb'd them spest the more
Than honest the streating throude's a meril: and

Temperature: 1.0
------------------------------------------------------------
Now is the winter of our discontent are queen.

KING RICHARD III:
Why, what doth you are heaven almost?

PRINCE EDWARD:
No more.
Wherefore doiery this time to fond at the pass
Whereof? Come, my bold to the good
Petroop's head: 'tis a m

Temperature: 1.5
------------------------------------------------------------
Now is the winter of our discontent are:
These being was fitter ten
you-rove-keep'd, for habiloya's may confly:

GREsso1OH OF KINGHAY:
Let's my petty Bolingbroke-book'd, hast I not upon been my base
Enforce-golous Rome less; on't, o'en

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
My lord, were I wont to thy heavens and
The first confess
Wide in the whomen the crown'd
Which the wounded bent them speak.

Second Murderer:
Why so trous Bolingbroke? what he is n

Temperature: 1.0
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
My lord, welcome unto young him;
So as it the most now; and he is gummentiness.

LEONTES:
Welcome, what for father.

PRINCE EDWARD:
Well, our grace:
I have depend thee wholesom to 

Temperature: 1.5
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
Richmondroes
Prown lady! spil'd, plant. Walk not pluck'd perns,
What Abas gentleman to you make these pace:
Lord, you sure!' I told tell hobed apt with away:
The spirit; on't, poor


Observations:

At 0.5, the model produces more predictable sequences and tends to remain
closer to common patterns found in the training data. Character names,
dialogue formatting, and Shakespearean vocabulary occur frequently.
However, the text can still lack semantic coherence.

At 1.0, the generated text becomes more varied while still retaining many
characteristics of Shakespearean dialogue. The model produces a wider range
of character names and sentence structures, but also introduces more unusual
and grammatically incorrect sequences.

At 1.5, the output becomes considerably more random. The model starts
producing malformed words and character names such as "GREsso1OH OF KINGHAY"
and "Richmondroes". This demonstrates how higher temperatures make less
probable character choices more likely.

The experiment also shows that temperature does not improve the underlying
model. It only changes how the model's predicted probability distribution
is sampled. Lower temperatures favour high-probability predictions, while
higher temperatures allow lower-probability predictions to be selected more
frequently.
"""