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



Observations:

The baseline model is able to generate text that exhibits several 
characteristics of the Shakespeare training corpus. It frequently produces
character names followed by a colon and dialogue, suggesting that it has
learned the structural format of the source text. It also generates
Shakespearean vocabulary and recurring grammatical constructions such as
"thou", "thee", "hither", and "shall".

The generated sentences are often locally grammatical and resemble
Shakespearean dialogue, but they frequently lack semantic coherence. The model
therefore appears to have learned many local syntactic and stylistic patterns,
while struggling to maintain coherent meaning over longer sequences.

The experiment also shows that the structure and specificity of the prompt
influence the generated text. Prompts containing a character name and an
existing piece of dialogue tend to produce more strongly structured
Shakespearean-style continuations.
"""