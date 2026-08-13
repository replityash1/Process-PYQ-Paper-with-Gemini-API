import json
import os
import sys
import time
from typing import Literal

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.1-flash-lite"

MAX_ATTEMPTS = 7
RETRY_BASE_SECONDS = 8
RETRY_MAX_SECONDS = 60
FINAL_SLEEP_SECONDS = 8

OUTPUT_DIR = "output_temp"
PAGES_DIR = "pages"


# ============================================================
# SCHEMAS
# ============================================================

class Option(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]

    hi: str = Field(
        description=(
            "Exact visible Hindi text of this official exam option. "
            "Do not translate, paraphrase, correct, or invent."
        )
    )

    en: str = Field(
        description=(
            "Exact visible English text of this official exam option. "
            "Do not translate, paraphrase, correct, or invent."
        )
    )


class OptionRationale(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]

    is_correct: bool = Field(
        description=(
            "True only if this option is the correct answer to the "
            "original exam question. Option 5 is the official "
            "'Question not attempted' OMR choice."
        )
    )

    rationale_hi: str
    rationale_en: str


class QuestionVariantOption(BaseModel):
    label: Literal["1", "2", "3", "4"]
    hi: str
    en: str


class QuestionVariant(BaseModel):
    question_hi: str
    question_en: str

    options: list[QuestionVariantOption] = Field(
        description="Exactly four normal answer choices."
    )

    correct_label: Literal["1", "2", "3", "4"]

    explanation_en: str


class QuestionMetadata(BaseModel):

    subject: str
    branch: str
    subtopic: str

    archetype: Literal[
        "Direct Fact",
        "Numerical/Calculative",
        "Assertion-Reasoning",
        "Statement-Based",
        "Matrix Matching",
        "Diagram/Visual",
        "Table-Based",
        "Other",
    ]

    has_translation_discrepancy: bool

    discrepancy_note: str | None = None

    difficulty: Literal[
        "Easy",
        "Medium",
        "Hard",
    ]

    blooms_level: Literal[
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
    ]

    keywords: list[str]

    estimated_seconds: int

    conceptual_trap_hi: str | None = None
    conceptual_trap_en: str | None = None

    prerequisite_topic: str | None = None

    ncert_mapping: str | None = None

    formula_law_dependency: str | None = None

    shortcut_hack_hi: str | None = None
    shortcut_hack_en: str | None = None

    clone_variant: QuestionVariant


class Question(BaseModel):

    source_page: int

    number: int

    question_hi: str
    question_en: str

    has_visuals: bool

    box_hi: list[int] | None = None
    box_en: list[int] | None = None

    image_hi: str | None = None
    image_en: str | None = None

    options: list[Option] = Field(
        description=(
            "Exactly FIVE official exam options. "
            "Options 1-4 are normal choices. "
            "Option 5 is the compulsory 'Question not attempted' "
            "OMR option."
        )
    )

    explanation_hi: str
    explanation_en: str

    option_rationales: list[OptionRationale]

    metadata: QuestionMetadata


class QuestionBank(BaseModel):
    questions: list[Question]


# ============================================================
# PROMPT
# ============================================================

EXTRACTION_PROMPT = r"""
You are an expert multimodal competitive-exam paper extraction system.

You are processing ONE rendered page from an RPSC examination paper.

Your first responsibility is EXACT TRANSCRIPTION.
Your second responsibility is QUESTION ANALYSIS.

Never sacrifice source fidelity to make the output look complete.

============================================================
OFFICIAL EXAM OPTION FORMAT
============================================================

THIS EXAM HAS FIVE OFFICIAL OPTIONS.

1 = normal answer choice
2 = normal answer choice
3 = normal answer choice
4 = normal answer choice
5 = Question not attempted

Option 5 is a genuine official examination/OMR option.

DO NOT delete option 5.

DO NOT replace option 5.

DO NOT create option 5 yourself after extraction.

Extract the actual visible wording of option 5.

============================================================
EXACT TRANSCRIPTION RULES
============================================================

Extract EVERY multiple-choice question visibly present on this page.

Preserve exactly:

- original printed question number
- Hindi wording
- English wording
- punctuation
- option order
- numbers
- units
- mathematical notation
- scientific notation
- negative signs
- fractions
- percentages
- statement numbering
- table/matrix relationships

Do not:

- paraphrase
- summarize
- improve grammar
- correct spelling
- silently fix printing errors
- invent missing words
- invent missing options
- use outside knowledge to reconstruct unclear text

The PAGE IMAGE is the source of truth.

============================================================
UNCLEAR / ILLEGIBLE TEXT
============================================================

If text is visibly present but cannot be reliably read, write:

[UNCLEAR]

If a region is present but genuinely illegible, write:

[ILLEGIBLE]

NEVER guess missing text because the question looks familiar.

============================================================
HINDI AND ENGLISH
============================================================

Extract Hindi and English independently.

Do not translate one language into the other.

Do not use one language to silently repair the other.

If the Hindi and English versions materially differ in meaning:

has_translation_discrepancy = true

and explain why.

============================================================
STATEMENT-BASED QUESTIONS
============================================================

Preserve every statement separately.

For example:

Statement I
Statement II
Statement III
Statement IV

must remain four statements.

Do not summarize them.

============================================================
MATHEMATICS / SCIENCE
============================================================

Preserve numerical values exactly.

Use LaTeX when appropriate:

$...$

and

$$...$$

Do not change numbers.

10^-3 must not become 10^-2.

H2SO4 must not become H2SO3.

1/2 must not become 12.

============================================================
VISUAL QUESTIONS
============================================================

Set has_visuals = true when the question materially depends on:

- diagram
- circuit
- graph
- map
- chemical structure
- biological figure
- table
- matrix
- flowchart
- complex visual layout

Even when has_visuals = true:

YOU MUST STILL EXTRACT ALL READABLE TEXT.

Do not delete readable options.

If the visual is important, provide:

[ymin, xmin, ymax, xmax]

Coordinates are normalized 0-1000 over the COMPLETE PAGE IMAGE.

The crop should include:

- question number
- question text
- relevant visual
- all five official options

============================================================
QUESTIONS SPANNING PAGES
============================================================

If a question is partially visible:

extract only what is actually visible.

Do not invent the continuation.

============================================================
SOLVING
============================================================

After extraction, solve the ORIGINAL question.

Use the extracted wording, not an imagined corrected version.

Provide:

- Hindi explanation
- English explanation
- rationale for every option 1-5

Option 5 is the official "Question not attempted" option.
It is not a normal knowledge option.

============================================================
METADATA
============================================================

Be conservative.

Use null when the information cannot be established confidently.

Do not invent:

- NCERT mappings
- formulas
- prerequisites
- shortcuts
- mnemonics
- conceptual traps

just to fill a field.

============================================================
BLOOM
============================================================

Use:

Remember
Understand
Apply
Analyze
Evaluate

Do not use Create for an ordinary MCQ.

============================================================
PRACTICE CLONE
============================================================

Generate a NEW question testing the same underlying concept.

The clone must:

- be independently solvable
- use different wording and/or numbers
- have exactly 4 normal options
- have exactly one correct answer
- have an explanation

The clone must NOT contain the exam's option 5.

============================================================
FINAL CHECK
============================================================

Before returning JSON verify:

1. Every visible question was extracted.
2. Every original question has exactly 5 official options.
3. Labels are exactly 1,2,3,4,5.
4. Option 5 was preserved.
5. No source text was invented.
6. Hindi and English were not silently translated.
7. Numbers and formulas were preserved.
8. Statement-based questions remain intact.
9. Visual questions retain readable text.
10. Every rationale matches its option.
11. Clone contains exactly four options.
12. Metadata uses null when appropriate.

Return ONLY JSON matching the provided schema.
"""


# ============================================================
# RETRY
# ============================================================

def is_permanent_error(exc: Exception) -> bool:
    text = str(exc).lower()

    permanent_markers = [
        "authentication",
        "unauthorized",
        "api key",
        "permission denied",
        "invalid api key",
        "invalid argument",
        "invalid request",
    ]

    return any(
        marker in text
        for marker in permanent_markers
    )


def call_gemini_with_retry(
    client,
    image_bytes: bytes,
):

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):

        try:

            print(
                f"Gemini attempt "
                f"{attempt}/{MAX_ATTEMPTS}"
            )

            return client.models.generate_content(
                model=MODEL_NAME,

                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    EXTRACTION_PROMPT,
                ],

                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuestionBank,
                ),
            )

        except Exception as exc:

            last_error = exc

            print(
                f"Gemini attempt {attempt} failed:"
            )

            print(exc)

            if is_permanent_error(exc):
                print(
                    "Permanent API error detected; "
                    "stopping retries."
                )
                raise

            if attempt == MAX_ATTEMPTS:
                break

            delay = min(
                RETRY_BASE_SECONDS * (
                    2 ** (attempt - 1)
                ),
                RETRY_MAX_SECONDS,
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini request failed after "
        f"{MAX_ATTEMPTS} attempts."
    ) from last_error


# ============================================================
# VALIDATION
# ============================================================

def validate_box(
    box,
    name: str,
):

    if box is None:
        return

    if (
        not isinstance(box, list)
        or len(box) != 4
    ):
        raise ValueError(
            f"{name} must be "
            "[ymin,xmin,ymax,xmax]."
        )

    if not all(
        isinstance(x, int)
        for x in box
    ):
        raise ValueError(
            f"{name} must contain integers."
        )

    ymin, xmin, ymax, xmax = box

    if not all(
        0 <= x <= 1000
        for x in box
    ):
        raise ValueError(
            f"{name} coordinates must "
            "be between 0 and 1000."
        )

    if ymax <= ymin:
        raise ValueError(
            f"{name}: ymax <= ymin."
        )

    if xmax <= xmin:
        raise ValueError(
            f"{name}: xmax <= xmin."
        )


def validate_question_bank(
    bank: QuestionBank,
):

    questions_by_number = {}

    for question in bank.questions:

        if question.number <= 0:
            raise ValueError(
                f"Invalid question number: "
                f"{question.number}"
            )

        if question.number in questions_by_number:
            raise ValueError(
                f"Duplicate question number "
                f"{question.number} on page "
                f"{question.source_page}"
            )

        questions_by_number[
            question.number
        ] = question

        if (
            not question.question_hi.strip()
            and not question.question_en.strip()
        ):
            raise ValueError(
                f"Question {question.number} "
                "has no text."
            )

        # ----------------------------------------------------
        # Five official options
        # ----------------------------------------------------

        expected_labels = [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

        if len(question.options) != 5:
            raise ValueError(
                f"Question {question.number} "
                "does not have exactly "
                "five options."
            )

        actual_labels = [
            option.label
            for option in question.options
        ]

        if actual_labels != expected_labels:
            raise ValueError(
                f"Question {question.number} "
                f"has invalid option labels: "
                f"{actual_labels}"
            )

        # ----------------------------------------------------
        # Rationales
        # ----------------------------------------------------

        if len(
            question.option_rationales
        ) != 5:
            raise ValueError(
                f"Question {question.number} "
                "does not have five rationales."
            )

        rationale_labels = [
            r.label
            for r in question.option_rationales
        ]

        if rationale_labels != expected_labels:
            raise ValueError(
                f"Question {question.number} "
                "has invalid rationale labels."
            )

        # ----------------------------------------------------
        # Boxes
        # ----------------------------------------------------

        validate_box(
            question.box_hi,
            f"Question {question.number} box_hi",
        )

        validate_box(
            question.box_en,
            f"Question {question.number} box_en",
        )

        # ----------------------------------------------------
        # Clone
        # ----------------------------------------------------

        clone = (
            question.metadata.clone_variant
        )

        if len(clone.options) != 4:
            raise ValueError(
                f"Question {question.number} "
                "clone must have exactly "
                "four options."
            )

        clone_labels = [
            option.label
            for option in clone.options
        ]

        if clone_labels != [
            "1",
            "2",
            "3",
            "4",
        ]:
            raise ValueError(
                f"Question {question.number} "
                "clone labels are invalid."
            )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        if not (
            5
            <= question.metadata.estimated_seconds
            <= 1800
        ):
            raise ValueError(
                f"Question {question.number} "
                "has unreasonable solving time."
            )


# ============================================================
# IMAGE CROPPING
# ============================================================

def crop_box(
    image: Image.Image,
    box: list[int] | None,
    destination: str,
) -> bool:

    if box is None:
        return False

    validate_box(
        box,
        "crop_box",
    )

    image_width, image_height = (
        image.size
    )

    ymin, xmin, ymax, xmax = box

    left = (
        xmin / 1000
    ) * image_width

    top = (
        ymin / 1000
    ) * image_height

    right = (
        xmax / 1000
    ) * image_width

    bottom = (
        ymax / 1000
    ) * image_height

    padding = 12

    left = max(
        0,
        left - padding,
    )

    top = max(
        0,
        top - padding,
    )

    right = min(
        image_width,
        right + padding,
    )

    bottom = min(
        image_height,
        bottom + padding,
    )

    if (
        right <= left
        or bottom <= top
    ):
        return False

    width = right - left
    height = bottom - top

    if (
        width < 25
        or height < 25
    ):
        return False

    if height > (
        image_height * 0.95
    ):
        return False

    cropped = image.crop(
        (
            int(left),
            int(top),
            int(right),
            int(bottom),
        )
    )

    cropped.save(
        destination,
        format="JPEG",
        quality=95,
        optimize=True,
    )

    return True


def create_visual_crops(
    page_number: int,
    bank: QuestionBank,
):

    page_path = (
        os.path.join(
            PAGES_DIR,
            f"page_{page_number}.jpg",
        )
    )

    if not os.path.exists(
        page_path
    ):
        print(
            f"WARNING: page image "
            f"not found: {page_path}"
        )
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    image = Image.open(
        page_path
    )

    for question in bank.questions:

        if not question.has_visuals:
            continue

        if question.box_hi:

            filename = (
                f"page{page_number}"
                f"_q{question.number}"
                f"_hi.jpg"
            )

            destination = (
                os.path.join(
                    OUTPUT_DIR,
                    filename,
                )
            )

            if crop_box(
                image,
                question.box_hi,
                destination,
            ):
                question.image_hi = (
                    filename
                )

        if question.box_en:

            filename = (
                f"page{page_number}"
                f"_q{question.number}"
                f"_en.jpg"
            )

            destination = (
                os.path.join(
                    OUTPUT_DIR,
                    filename,
                )
            )

            if crop_box(
                image,
                question.box_en,
                destination,
            ):
                question.image_en = (
                    filename
                )


# ============================================================
# RESPONSE PARSER
# ============================================================

def parse_response(
    response_text: str,
) -> QuestionBank:

    if not response_text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned = response_text.strip()

    if cleaned.startswith(
        "```"
    ):

        lines = (
            cleaned.splitlines()
        )

        if (
            lines
            and lines[0].strip().startswith(
                "```"
            )
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    try:

        bank = (
            QuestionBank.model_validate_json(
                cleaned
            )
        )

    except ValidationError as exc:

        raise ValueError(
            "Gemini output failed schema validation:\n"
            f"{exc}"
        ) from exc

    validate_question_bank(
        bank
    )

    return bank


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:
        print(
            "Usage: "
            "python3 scripts/extract_api.py <page>"
        )
        sys.exit(1)

    try:
        page_number = int(
            sys.argv[1]
        )
    except ValueError:
        print(
            "ERROR: Page number must be an integer."
        )
        sys.exit(1)

    page_path = (
        os.path.join(
            PAGES_DIR,
            f"page_{page_number}.jpg",
        )
    )

    output_path = (
        os.path.join(
            OUTPUT_DIR,
            f"page_{page_number}.json",
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        page_path
    ):
        print(
            f"ERROR: Page image does not exist:"
            f" {page_path}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        with open(
            page_path,
            "rb",
        ) as file:

            image_bytes = (
                file.read()
            )

    except OSError as exc:

        print(
            f"ERROR reading image: {exc}"
        )
        sys.exit(1)

    if not image_bytes:
        print(
            "ERROR: Page image is empty."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )

    print(
        "======================================"
    )

    print(
        f"PROCESSING PAGE {page_number}"
    )

    print(
        f"MODEL: {MODEL_NAME}"
    )

    print(
        "======================================"
    )

    try:

        response = (
            call_gemini_with_retry(
                client,
                image_bytes,
            )
        )

    except Exception as exc:

        print(
            f"ERROR: Gemini failed:\n{exc}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    try:

        bank = parse_response(
            response.text
        )

    except Exception as exc:

        print(
            f"ERROR: Output validation failed:\n{exc}"
        )

        raw_path = (
            os.path.join(
                OUTPUT_DIR,
                f"page_{page_number}_raw.txt",
            )
        )

        with open(
            raw_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                response.text
                if response.text
                else "<EMPTY>"
            )

        print(
            f"Raw response saved to {raw_path}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Source page is authoritative from worker
    # --------------------------------------------------------

    for question in bank.questions:
        question.source_page = (
            page_number
        )

    # --------------------------------------------------------
    # Create crops
    # --------------------------------------------------------

    try:

        create_visual_crops(
            page_number,
            bank,
        )

    except Exception as exc:

        print(
            f"ERROR creating image crops: {exc}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Serialize
    # --------------------------------------------------------

    data = bank.model_dump(
        mode="json"
    )

    for question in data["questions"]:

        # Bounding boxes are intermediate data.
        question.pop(
            "box_hi",
            None,
        )

        question.pop(
            "box_en",
            None,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

    except OSError as exc:

        print(
            f"ERROR saving output: {exc}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    visual_count = sum(
        1
        for q in bank.questions
        if q.has_visuals
    )

    print()
    print(
        "SUCCESS"
    )

    print(
        f"Page: {page_number}"
    )

    print(
        f"Questions: {len(bank.questions)}"
    )

    print(
        f"Visual questions: {visual_count}"
    )

    print(
        f"Saved: {output_path}"
    )

    time.sleep(
        FINAL_SLEEP_SECONDS
    )


if __name__ == "__main__":
    main()
