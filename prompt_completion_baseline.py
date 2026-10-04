from model.gpt import GPT
from pathlib import Path
from tokenization.tokenizer import Tokenizer

from generation.prompts import SHAKESPEARE_PROMPTS
from generation.utils import generate_completions


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "check_points" / "baseline.pt"
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

KING RICHARD III:
The worst of thou that wilt welcome hither.

CLARENCE:
Where is the content?

DUKE OF YORK:
Why should he were he somets of the content?

DUKE OF YORK:
Why should he were he somets

============================================================
Prompt 2
============================================================
Now is the winter of our discontent

Completion:
Now is the winter of our discontent.

EDWARD:
No blood a thousand work, what he should be,
And the softer of the seat of them for an
The senses of the dead of the deadly of thee.

CLAUDIO:
The should be the contents of the world to the

============================================================
Prompt 3
============================================================
The night is dark

Completion:
The night is darken of England?
Who shall hath he would have these wounds the seas of the bed
The senses of the world that we heart follow them.

Clown:
Why so? then, then, I have sense that he would be som.

CLARENCE

============================================================
Prompt 4
============================================================
To be, or not to be, that is the question:

Completion:
To be, or not to be, that is the question:
I say the news of the world's boldness that which he
Or else thou hast spoke of his force.

ANTIGONUS:
I think it for his prayers and the contrave:
The noble words weak and speech the world's eye,
Th

============================================================
Prompt 5
============================================================
Fie, fie, upon this quiet life! I want work.

Completion:
Fie, fie, upon this quiet life! I want work.

PRINCE EDWARD:
What say you hear? What say? I have here content?

Nurse:
To here the comes hither.

MENENIUS:
The worm and is the world that he would be so the
some the sun and the sun and honour of

============================================================
Prompt 6
============================================================
FIRST CITIZEN:
Before we proceed any further, hear me speak.

Completion:
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

============================================================
Prompt 7
============================================================
The king shall

Completion:
The king shall be the law to thee here?
Why, the crown make thee thee to speak what the earth
Of that to beat the coverty to the deputy.

CAMILLO:
I say thee, good sweet Montgarden, they shall not so.

ISABELLA:
Th

============================================================
Prompt 8
============================================================
Romeo:

Completion:
Romeo: sure to the content.

MENENIUS:
The common are too much other to thee worth
The sun she world as thou art seest, and the sea
dogs of the sense and the senate of them to the
banishment of the seat o'



Baseline Model Generation Report
================================

The baseline model was evaluated using several Shakespeare-style prompts.
The model produces recognizable character names, dialogue formatting,
Shakespearean vocabulary, and recurring grammatical patterns such as
"thou", "thee", "hither", and "shall".

Many generated passages are locally plausible and resemble Shakespearean
dialogue. However, longer sequences frequently lose semantic coherence and
contain repetitive or grammatically incorrect constructions.

An important consideration is that this baseline model was strongly
overfitted to the training data. It achieved a training loss of approximately
0.96, compared with a substantially higher validation loss of approximately
1.831. Therefore, some of the seemingly strong generation results may be
partly explained by the model fitting the training corpus very closely rather
than generalizing well to unseen text.

The model clearly learned useful local patterns from the dataset, but the
large gap between training and validation performance shows that these
patterns did not transfer equally well to unseen data.

Overall, the baseline demonstrates that the model can learn recognizable
Shakespearean structure and style, but its generation quality should not be
interpreted independently of its overfitting. This provided an important
motivation for introducing regularization and evaluating models primarily
using validation performance rather than training loss alone.
"""