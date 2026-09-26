from pathlib import Path

import torch

from generation.utils import generate_completions_with_ksampling_temp
from model.gpt import GPT
from tokenization.tokenizer import Tokenizer


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "baseline.pt"
TOKENIZER_PATH = ROOT / "data" / "shakespeare_char_tokenizer_config.pt"

K_VALUES = [1, 5, 10, 20]
TEMPERATURE = 1.0
NEW_TOKENS = 200

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

        for k in K_VALUES:
            torch.manual_seed(22)

            completion = generate_completions_with_ksampling_temp(
                model=model,
                prompts=[prompt],
                tokenizer=tokenizer,
                new_tokens=NEW_TOKENS,
                temperature=TEMPERATURE,
                k=k,
            )[0]

            print(f"K: {k}")
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

K: 1
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

KING RICHARD III:
The worst of thou that wilt welcome hither.

CLARENCE:
Where is the content?

DUKE OF YORK:
Why should he were he somets of the content?

DUKE OF YORK:
Why should he were he somets

K: 5
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

HENRY BOLING EDWARD ISHY:
Was is there embol, holds it so; that thou art
Where is the mother wholesome bold, his soul tongue
May stones and and she his power?'

Second Murderer:
Thou hast son my hea

K: 10
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

HENRY BOLING EDWARD ISHY:
Was is my deny should have been borned most
A much thought of me on house of your
neith; I will accomen? come.
But is none for out to marketh, which thou not shouldst
In sp

K: 20
------------------------------------------------------------
KING HENRY:
What news from the north, good my lord?

HENRY BOLING EDWARD ISHHENSS:
What ever woman?

JULIET:
And let me should be that I bear, both:
He contents aften my war hath been consent;
The nenext here's perMed at God's he wound,
Mork'd heaven 

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

K: 1
------------------------------------------------------------
Now is the winter of our discontent.

EDWARD:
No blood a thousand work, what he should be,
And the softer of the seat of them for an
The senses of the dead of the deadly of thee.

CLAUDIO:
The should be the contents of the world to the

K: 5
------------------------------------------------------------
Now is the winter of our discontent?
O bloody party, the truth is the speen, heaven follows;
Anger for such freedoms as they
from his face, to say their hearts, for the
shoes humourning to thee to black thy heart.

LADY CAPULET:
Why so

K: 10
------------------------------------------------------------
Now is the winter of our discontent?
How camest adon after thou art thou service.

DUKE OF AUMERLE:
What satisforce and I be, so advaintance
Have no light to can him, for such; and none forer?

GLOUCESTER:
You know now the descent have

K: 20
------------------------------------------------------------
Now is the winter of our discontent?
How camest adon after thou art thou service.

DUKE OF AUMERLE:
What satisforce and I beht thee?

LUCIO:
Hark he thoughts, and betray does he illess him?

ANGELO:
Good lords and pass the descene?
How

============================================================
Prompt 3
============================================================
FIRST CITIZEN:
Before we proceed any further, hear me speak.

K: 1
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

MENENIUS:
I am sad heart to thee.

SICINIUS:
Why, sir;
The consument I have auth.

CORIOLANUS:
Well, he will have me to the crown?

BRUTUS:
Ay, to say you are princely the world be son?

CORIOLANUS:

K: 5
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

HENRY BOLINGBROKE:
We shall he her heart of her hence; how, ho!

GLOUCESTER:
What do you, that honour'd worthy affliction here,
Sir thou be sometime to thee to be somety?

CAMILLO:
And he short, his

K: 10
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

HENRY BOLINGBROKE:
We shall he her heart of her hence; so both,
The son ax empty, that hour'd holds
With controp in this cannot.

ROMEO:
Marshalt I cannot put to give him home.

SLERS:
May by him a 

K: 20
------------------------------------------------------------
FIRST CITIZEN:
Before we proceed any further, hear me speak.

HENRY BOLINGBROKE:
We shall I do, not gentle thine of guile
At our hearts with all the house of Hereford cloak?

RICHMOND:
Plebeis consent folly and fury go to mone
My dearted confound he should him


Observations:

The experiment shows that the value of k has a noticeable effect on the
generated text. With a low value of k, such as k=1, the model produces more
restricted and locally plausible continuations. The generated text generally
maintains a recognizable Shakespearean structure, including character names,
dialogue, and vocabulary similar to the training corpus.

As k increases, the generated text becomes less constrained and shows greater
variation in the selected tokens. However, this increased freedom is also
accompanied by a gradual reduction in grammatical and semantic coherence. For
example, larger values of k produce increasingly unusual character names,
fragmented sentences, and combinations of words that do not form meaningful
phrases.

The effect is visible across all three prompts. At k=1, the generations tend
to preserve the structural and stylistic characteristics of the Shakespeare
corpus more consistently, while at k=10 and k=20, the model produces more
irregular and unpredictable continuations.
"""