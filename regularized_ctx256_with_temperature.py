from pathlib import Path

import torch

from generation.utils import generate_completions_with_temp
from model.gpt import GPT
from tokenization.tokenizer import Tokenizer


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "regularized_ctx256.pt"
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
============================================================
Prompt 1
============================================================
KING HENRY:
What news from the north, good my lord?

Temperature: 0.5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD IIIII:
God come thy good my lord, I must be obey'd:
The name of my mother,
Whom they have done and the man of her mark,
And then bear a blessed as with his hot
To the state and acceptai

Temperature: 1.0
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD IIIIII:
Not or else honours to you first absolute,
As I not thee lay to the King of York,
As as my untimed paces for the posts:
Standols tear hows with his hot my true pleasants
Slept-u

Temperature: 1.5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD IIIIII:
Not or else, wholest is he first absolachers?
O Hy, thou lay 1 great: Thee stole too,
I meturness to forbear': you sultle I moat ten my
behort I serve look Rome your porth, paac

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Temperature: 0.5
------------------------------------------------------------
Now is the winter of our discontent
To fair prove and the virtue enemy wind.

GLOUCESTER:
And that I may come to the more than the state
Than you may be sworn and for the sun.

GLOUCESTER:
I will have such hath person to him,
And so be

Temperature: 1.0
------------------------------------------------------------
Now is the winter of our discontent are:
There we did not it commend to-night!

MENENIUS:
O much, sended curfsiders!

CORIOLANUS:
Methink you, and good lord?

Second Citizen:
It resump home.

COMINIUS:
Bid he so must speak with him a-d

Temperature: 1.5
------------------------------------------------------------
Now is the winter of our discontent are:
There's advow, yet by, men
youwn lady is, have zecamiolation you namptuces;
mo1 great: These come too, I must tell him here;
Call assultness moathse went short and beats
Pet: is your post! Woo-m

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
Why, thou shalt we have been a man,
And be sented come to come on the best,
And the duke of the true of meet of the suspect
To the bear of the strong of the steels of man
And so be

Temperature: 1.0
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
I cannot me;
No wind by thy shame hand.
Was ever church acceptuary 'Which is they frame,
And there would have true. The story unplucker must
That I shall love Rome less; that I kno

Temperature: 1.5
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

KING RICHARD III:
I come or else, wholesomes,
On my hcab.
Was ever ccurity;
He succes my King Ladius From God and tell have himself,
Fouststly I moat ten my beholdier
I hatroy their less; or have a 


Generation comparison across context lengths:
1) Current context length is 256
2) The previous model used a context length of 128

The 256-context model produces noticeably different generations from the
128-context model when temperature sampling is used. The longer context
allows the model to condition its predictions on a larger portion of the
preceding text, resulting in somewhat more developed dialogue structures and
more varied continuations.

At a temperature of 0.5, the 256-context model generally produces longer
and more structured passages before becoming incoherent. It frequently
maintains a recognizable dialogue format, with character names followed by
several lines of text. The 128-context model also produces recognizable
dialogue, but its continuations tend to transition between characters and
phrases more abruptly.

At a temperature of 1.0, the 256-context model continues to generate varied
Shakespearean-style dialogue while maintaining more consistent local
structure. The 128-context model produces similarly varied text, but more
frequently jumps between unrelated characters and contains abrupt or
fragmented transitions.

At a temperature of 1.5, both models become increasingly noisy and produce
malformed words and unusual character names. However, the 256-context model
still maintains recognizable dialogue structures in several generations,
whereas the 128-context model more quickly produces highly fragmented text.

Overall, increasing the context length from 128 to 256 does not eliminate the
grammatical and semantic problems of the model, nor does it clearly produce
more coherent text in every generation. Its main observable effect is that
the model can maintain larger-scale dialogue structures and more varied
continuations before losing coherence.

The difference is most noticeable when sampling at lower temperatures,
where the 256-context model appears better able to sustain a continuation
over a longer sequence. At higher temperatures, the additional context is
less apparent because the increased sampling randomness dominates the
generation behaviour.

These results suggest that the longer context provides the model with more
information for generation, but the current model has not yet learned to
consistently exploit that information for long-range semantic coherence.
"""