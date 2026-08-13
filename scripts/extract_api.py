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

MODEL_NAME = "gemini-3.5-flash-lite"

MAX_ATTEMPTS = 7
RETRY_BASE_SECONDS = 8
RETRY_MAX_SECONDS = 60
FINAL_SLEEP_SECONDS = 12

OUTPUT_DIR = "output_temp"
PAGES_DIR = "pages"


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class Option(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]

    hi: str = Field(
        description=(
            "Exact visible Hindi text of this option. "
            "Do not translate, paraphrase, correct, or invent text. "
            "Use empty string only when the option is entirely visual."
        )
    )

    en: str = Field(
        description=(
            "Exact visible English text of this option. "
            "Do not translate, paraphrase, correct, or invent text. "
            "Use empty string only when the option is entirely visual."
        )
    )


class OptionRationale(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]

    is_correct: bool = Field(
        description=(
            "True only when this option is the correct answer to the "
            "original question. Option 5 is the compulsory OMR "
            "'Question not attempted' choice and should not be treated "
            "as a normal knowledge answer."
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

    # Internal sequential ID assigned later by merge script.
    sequence: int | None = None

    question_hi: str
    question_en: str

    has_visuals: bool

    box_hi: list[int] | None = None
    box_en: list[int] | None = None

    image_hi: str | None = None
    image_en: str | None = None

    options: list[Option] = Field(
        description=(
            "Exactly FIVE OFFICIAL exam options. "
            "Options 1-4 are the normal answer choices. "
            "Option 5 is the compulsory 'Question not attempted' "
            "OMR choice used by this exam format."
        )
    )

    explanation_hi: str
    explanation_en: str

    option_rationales: list[OptionRationale]

    metadata: QuestionMetadata


class QuestionBank(BaseModel):
    questions: list[Question]


# ============================================================
# EXTRACTION PROMPT
# ============================================================

EXTRACTION_PROMPT = r"""
You are an expert multimodal competitive-exam paper extraction system.

You are processing ONE rendered page from an RPSC examination paper.

Your job has two priorities:

PRIMARY:
Faithfully extract what is visibly present in the image.

SECONDARY:
Solve the extracted questions and generate educational metadata.

Never sacrifice transcription accuracy for completeness.

============================================================
IMPORTANT EXAM FORMAT
============================================================

This examination uses FIVE OFFICIAL OPTIONS.

Option 1 = normal answer choice.
Option 2 = normal answer choice.
Option 3 = normal answer choice.
Option 4 = normal answer choice.
Option 5 = "Question not attempted".

Option 5 is a REAL, OFFICIAL EXAMINATION/OMR OPTION.

It MUST be preserved.

Do NOT delete option 5.

Do NOT replace option 5 after extraction.

Do NOT treat option 5 as an application-only/UI field.

Extract exactly what is visibly printed for option 5.

============================================================
EXACT TRANSCRIPTION
============================================================

1. Extract EVERY MCQ visibly present on this page.

2. Preserve the original:
   - question number
   - Hindi wording
   - English wording
   - option order
   - punctuation
   - numbers
   - units
   - mathematical symbols
   - fractions
   - percentages
   - negative signs
   - scientific notation
   - statement numbering
   - table structure

3. DO NOT improve spelling.

4. DO NOT correct grammar.

5. DO NOT silently correct printing errors.

6. DO NOT paraphrase.

7. DO NOT summarize.

8. DO NOT translate Hindi into English.

9. DO NOT translate English into Hindi.

10. NEVER invent text that is not visible.

============================================================
UNCLEAR TEXT
============================================================

If text is present but genuinely unreadable, use:

[UNCLEAR]

If a portion is visibly present but illegible, use:

[ILLEGIBLE]

Do NOT guess the missing content from subject knowledge.

Do NOT reconstruct a question because you recognize a common exam pattern.

Do NOT assume a familiar formula or wording.

The source image is the authority.

============================================================
HINDI + ENGLISH
============================================================

If both Hindi and English versions are visible, extract both independently.

Do not use one language to silently repair the other.

If the two versions materially differ in meaning, set:

has_translation_discrepancy = true

and explain the exact discrepancy.

============================================================
QUESTION STRUCTURE
============================================================

Preserve statement-based questions exactly.

For example, if the source contains:

Statement I
Statement II
Statement III
Statement IV

keep all four statements.

Do NOT turn them into a summary.

If options refer to combinations such as:

1. Only I
2. Only II
3. I and II
4. II and III
5. Question not attempted

preserve those choices exactly.

============================================================
MATHEMATICS / SCIENCE
============================================================

Preserve mathematical and scientific notation.

Use LaTeX where useful:

Inline:
$...$

Display:
$$...$$

Do not change numerical values.

For example:

10^-3 MUST NOT become 10^-2.

H2SO4 MUST NOT become H2SO3.

1/2 MUST NOT become 12.

============================================================
VISUAL QUESTIONS
============================================================

Set has_visuals = true when the question materially depends on:

- diagram
- graph
- map
- circuit
- chemical structure
- biological figure
- table
- matrix
- flowchart
- complex visual layout

IMPORTANT:

Having a visual does NOT mean the textual content should be removed.

Still extract all readable text.

If visual information is necessary and cannot be faithfully represented as
plain text, provide bounding boxes.

Bounding box format:

[ymin, xmin, ymax, xmax]

All coordinates are normalized from 0 to 1000 relative to the COMPLETE PAGE.

The relevant box should include:

- question number
- question text
- visual
- all five official options

============================================================
PAGE BOUNDARIES
============================================================

A question may start near the bottom of a page.

If only part of it is visible:

- extract only the visible content
- do not invent the missing continuation

If text clearly continues onto another page:

do not fabricate the missing portion.

============================================================
ANSWER / SOLUTION
============================================================

After faithful transcription, solve the original question independently.

Do not solve a guessed version.

Provide:

- explanation in Hindi
- explanation in English
- rationale for every option 1-5

Option rationales MUST correspond to the exact same option labels.

Option 5 represents "Question not attempted".

It is not a normal conceptual answer choice.

============================================================
METADATA
============================================================

Be conservative.

Use null when information cannot be established confidently.

Do NOT invent:

- NCERT chapter references
- formulas
- prerequisite topics
- shortcuts
- mnemonics
- conceptual traps

just to fill fields.

NCERT mapping should only be supplied when reasonably defensible.

============================================================
BLOOM'S TAXONOMY
============================================================

Classify the ORIGINAL MCQ using:

Remember
Understand
Apply
Analyze
Evaluate

Do NOT classify ordinary MCQs as Create.

============================================================
PRACTICE CLONE
============================================================

Generate one new practice question based on the same concept.

The clone must:

- test the same core concept
- be independently solvable
- change wording/numbers/conditions where appropriate
- contain exactly FOUR normal answer choices
- contain one unambiguous correct answer
- include an English explanation

IMPORTANT:

The generated clone is NOT part of the original exam.

Therefore it MUST NOT contain the original exam's fifth
"Question not attempted" option.

============================================================
FINAL SELF-CHECK
============================================================

Before returning JSON verify:

1. Every visible question was extracted.
2. Every question has exactly FIVE official options.
3. Option labels are exactly 1,2,3,4,5.
4. Option 5 was preserved.
5. No source text was invented.
6. Hindi and English were not silently translated.
7. Numbers and formulas were preserved.
8. Statement-based questions remain intact.
9. Visual questions still retain readable text.
10. Every rationale matches its option label.
11. Clone has exactly FOUR normal options.
12. Clone correct label is 1,2,3,or 4.
13. Metadata uses null where appropriate.

Return ONLY JSON matching the provided schema.
"""


# ============================================================
# HELPERS
# ============================================================

def is_retryable_exception(exc: Exception) -> bool:
    """
    Avoid wasting all retry attempts on obvious permanent errors.
    """
    text = str(exc).lower()

    permanent_markers = [
        "authentication",
        "unauthorized",
        "api key",
        "permission denied",
        "invalid api key",
        "not found",
        "invalid argument",
        "invalid request",
        "schema",
    ]

    return not any(
        marker in text
        for marker in permanent_markers
    )


def call_gemini_with_retry(
    client,
    image_bytes: bytes,
    prompt: str,
):
    last_exception = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            print(
                f"Gemini attempt {attempt}/{MAX_ATTEMPTS}"
            )

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuestionBank,
                ),
            )

        except Exception as exc:

            last_exception = exc

            print(
                f"Gemini attempt {attempt} failed: {exc}"
            )

            if not is_retryable_exception(exc):
                print(
                    "Failure appears permanent; "
                    "not retrying."
                )
                raise

            if attempt == MAX_ATTEMPTS:
                break

            delay = min(
                RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                RETRY_MAX_SECONDS,
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Gemini failed after {MAX_ATTEMPTS} attempts."
    ) from last_exception


def validate_box(
    box,
    field_name: str,
):
    if box is None:
        return

    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(
            f"{field_name} must be [ymin,xmin,ymax,xmax]."
        )

    if not all(
        isinstance(v, int)
        for v in box
    ):
        raise ValueError(
            f"{field_name} must contain integers."
        )

    ymin, xmin, ymax, xmax = box

    if not all(
        0 <= v <= 1000
        for v in box
    ):
        raise ValueError(
            f"{field_name} values must be 0-1000."
        )

    if ymax <= ymin:
        raise ValueError(
            f"{field_name}: ymax <= ymin."
        )

    if xmax <= xmin:
        raise ValueError(
            f"{field_name}: xmax <= xmin."
        )


def validate_question_bank(
    bank: QuestionBank,
):
    seen = set()

    for q in bank.questions:

        if q.number <= 0:
            raise ValueError(
                f"Invalid question number: {q.number}"
            )

        if q.number in seen:
            raise ValueError(
                f"Duplicate question number {q.number} "
                f"on page {q.source_page}"
            )

        seen.add(q.number)

        if not q.question_hi.strip() and not q.question_en.strip():
            raise ValueError(
                f"Question {q.number} has no text."
            )

        # ----------------------------------------------------
        # ORIGINAL EXAM OPTIONS
        # ----------------------------------------------------

        if len(q.options) != 5:
            raise ValueError(
                f"Question {q.number}: expected exactly "
                f"5 exam options, got {len(q.options)}."
            )

        expected = [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

        actual = [
            option.label
            for option in q.options
        ]

        if actual != expected:
            raise ValueError(
                f"Question {q.number}: invalid option sequence "
                f"{actual}."
            )

        # ----------------------------------------------------
        # RATIONALES
        # ----------------------------------------------------

        if len(q.option_rationales) != 5:
            raise ValueError(
                f"Question {q.number}: expected five rationales."
            )

        actual_rationale_labels = [
            r.label
            for r in q.option_rationales
        ]

        if actual_rationale_labels != expected:
            raise ValueError(
                f"Question {q.number}: invalid rationale sequence."
            )

        # ----------------------------------------------------
        # BOXES
        # ----------------------------------------------------

        validate_box(
            q.box_hi,
            f"Question {q.number} box_hi",
        )

        validate_box(
            q.box_en,
            f"Question {q.number} box_en",
        )

        # ----------------------------------------------------
        # CLONE
        # ----------------------------------------------------

        clone = q.metadata.clone_variant

        if len(clone.options) != 4:
            raise ValueError(
                f"Question {q.number}: clone must have "
                f"exactly four options."
            )

        clone_labels = [
            opt.label
            for opt in clone.options
        ]

        if clone_labels != [
            "1",
            "2",
            "3",
            "4",
        ]:
            raise ValueError(
                f"Question {q.number}: clone labels invalid."
            )

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if not (
            5
            <= q.metadata.estimated_seconds
            <= 1800
        ):
            raise ValueError(
                f"Question {q.number}: unreasonable "
                f"estimated_seconds={q.metadata.estimated_seconds}"
            )


# ============================================================
# IMAGE CROPPING
# ============================================================

def crop_box(
    img: Image.Image,
    box: list[int] | None,
    output_filepath: str,
) -> bool:

    if box is None:
        return False

    validate_box(
        box,
        "crop_box",
    )

    img_w, img_h = img.size

    ymin, xmin, ymax, xmax = box

    left = (
        xmin / 1000.0
    ) * img_w

    top = (
        ymin / 1000.0
    ) * img_h

    right = (
        xmax / 1000.0
    ) * img_w

    bottom = (
        ymax / 1000.0
    ) * img_h

    # Small padding prevents cutting labels/edges.
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
        img_w,
        right + padding,
    )

    bottom = min(
        img_h,
        bottom + padding,
    )

    if right <= left or bottom <= top:
        return False

    width = right - left
    height = bottom - top

    if width < 25 or height < 25:
        return False

    # Do not accept accidental whole-page boxes.
    if height > 0.95 * img_h:
        return False

    cropped = img.crop(
        (
            int(left),
            int(top),
            int(right),
            int(bottom),
        )
    )

    cropped.save(
        output_filepath,
        format="JPEG",
        quality=95,
        optimize=True,
    )

    return True


def process_visual_crops(
    page_num: int,
    bank: QuestionBank,
):
    image_path = os.path.join(
        PAGES_DIR,
        f"page_{page_num}.jpg",
    )

    if not os.path.exists(image_path):
        print(
            f"WARNING: image not found for crops: "
            f"{image_path}"
        )
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    img = Image.open(image_path)

    for q in bank.questions:

        if not q.has_visuals:
            continue

        if q.box_hi:

            filename = (
                f"page{page_num}_q{q.number}_hi.jpg"
            )

            destination = os.path.join(
                OUTPUT_DIR,
                filename,
            )

            if crop_box(
                img,
                q.box_hi,
                destination,
            ):
                q.image_hi = filename

        if q.box_en:

            filename = (
                f"page{page_num}_q{q.number}_en.jpg"
            )

            destination = os.path.join(
                OUTPUT_DIR,
                filename,
            )

            if crop_box(
                img,
                q.box_en,
                destination,
            ):
                q.image_en = filename


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_response(
    response_text: str,
) -> QuestionBank:

    if not response_text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    cleaned = response_text.strip()

    # Defensive handling in case a model/API wrapper returns
    # fenced JSON despite response_mime_type.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        bank = QuestionBank.model_validate_json(
            cleaned
        )

    except ValidationError as exc:
        raise ValueError(
            "Gemini output failed Pydantic validation:\n"
            f"{exc}"
        ) from exc

    validate_question_bank(bank)

    return bank


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python3 scripts/extract_api.py <page_number>"
        )
        sys.exit(1)

    try:
        page_num = int(sys.argv[1])
    except ValueError:
        print(
            f"ERROR: Invalid page number: {sys.argv[1]}"
        )
        sys.exit(1)

    image_path = os.path.join(
        PAGES_DIR,
        f"page_{page_num}.jpg",
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"page_{page_num}.json",
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Verify page
    # --------------------------------------------------------

    if not os.path.exists(image_path):
        print(
            f"ERROR: Page image not found: {image_path}"
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
            "ERROR: GEMINI_API_KEY is not configured."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        with open(
            image_path,
            "rb",
        ) as f:
            image_bytes = f.read()

    except OSError as exc:

        print(
            f"ERROR: Could not read {image_path}: {exc}"
        )
        sys.exit(1)

    if not image_bytes:
        print(
            f"ERROR: Empty page image: {image_path}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )

    print(
        "======================================"
    )
    print(
        f"PROCESSING PAGE {page_num}"
    )
    print(
        f"MODEL: {MODEL_NAME}"
    )
    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    try:

        response = call_gemini_with_retry(
            client,
            image_bytes,
            EXTRACTION_PROMPT,
        )

    except Exception as exc:

        print(
            f"ERROR: Gemini request failed: {exc}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Parse + validate
    # --------------------------------------------------------

    try:

        bank = parse_response(
            response.text
        )

    except Exception as exc:

        print(
            f"ERROR: Invalid Gemini output:\n{exc}"
        )

        raw_output = os.path.join(
            OUTPUT_DIR,
            f"page_{page_num}_raw.txt",
        )

        with open(
            raw_output,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                response.text
                if response.text
                else "<EMPTY>"
            )

        print(
            f"Raw response saved to: {raw_output}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Add page number explicitly
    # --------------------------------------------------------

    for q in bank.questions:
        q.source_page = page_num

    # --------------------------------------------------------
    # Generate visual crops
    # --------------------------------------------------------

    try:

        process_visual_crops(
            page_num,
            bank,
        )

    except Exception as exc:

        print(
            f"ERROR while generating visual crops: {exc}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Prepare final JSON
    # --------------------------------------------------------

    output_data = bank.model_dump(
        mode="json"
    )

    # Bounding boxes are only an intermediate extraction
    # mechanism. The final archive uses image_hi/image_en.
    for q in output_data["questions"]:
        q.pop("box_hi", None)
        q.pop("box_en", None)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output_data,
                f,
                ensure_ascii=False,
                indent=4,
            )

    except OSError as exc:

        print(
            f"ERROR: Could not save {output_path}: {exc}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(
        bank.questions
    )

    visuals = sum(
        1
        for q in bank.questions
        if q.has_visuals
    )

    print()
    print(
        "SUCCESS"
    )
    print(
        f"Page: {page_num}"
    )
    print(
        f"Questions: {total}"
    )
    print(
        f"Visual questions: {visuals}"
    )
    print(
        f"Saved: {output_path}"
    )

    time.sleep(
        FINAL_SLEEP_SECONDS
    )


if __name__ == "__main__":
    main()
