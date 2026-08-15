"""
RPSC Science PYQ extraction — page-level worker.

This script sends ONE rendered page image to Gemini and asks it to return
a structured, bilingual, SVG-based question extraction that matches the
"master prompt" schema (see EXTRACTION_PROMPT below). It intentionally
does NOT crop bounding boxes out of the page image for diagrams — per the
prompt, diagrams/figures/graphs/tables/matching-columns etc. must be
reconstructed as inline SVG (or left null with a note if reconstruction is
genuinely impossible), not represented as raster crops.

Because the master prompt describes a *paper-level* JSON object (with a
top-level "paper" block, "paper_statistics", and "validation"), and this
script only ever sees one page at a time, this worker only produces the
per-page "questions" array. A separate merge step (not included here)
should concatenate every page's "questions" array, fill in the shared
"paper" metadata once, and compute "paper_statistics" / "validation" over
the full set.
"""

import json
import os
import sys
import time
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

MAX_ATTEMPTS = 7
RETRY_BASE_SECONDS = 8
RETRY_MAX_SECONDS = 60
FINAL_SLEEP_SECONDS = 8

OUTPUT_DIR = "output_temp"
PAGES_DIR = "pages"


# ============================================================
# SCHEMA
# (mirrors the JSON structure described in the master prompt,
#  sections 6-25 and 32)
# ============================================================

QuestionType = Literal[
    "single_choice",
    "multiple_choice",
    "assertion_reason",
    "statement_based",
    "statement_correctness",
    "matching",
    "matrix_matching",
    "column_matching",
    "sequence_ordering",
    "true_false",
    "multiple_statements",
    "numerical",
    "diagram_based",
    "graph_based",
    "table_based",
    "image_based",
    "passage_based",
    "case_based",
    "formula_based",
    "fill_in_the_blank",
    "other",
]

ContentOriginValue = Literal["source", "reconstructed", "generated", "inferred"]


class BilingualText(BaseModel):
    english: str | None = None
    hindi: str | None = None


class Visual(BaseModel):
    type: Literal["svg"] = "svg"
    purpose: str | None = Field(
        default=None,
        description="e.g. question_diagram, option_diagram, graph, table_visual",
    )
    placement: str | None = Field(
        default=None,
        description="e.g. question_body, option_A, option_B",
    )
    svg: str = Field(
        description="A complete, self-contained, renderable SVG string. "
        "Must start with '<svg' and end with '</svg>'."
    )
    description: str = Field(
        description="Plain-text accessibility/search description of what the SVG shows."
    )
    source_fidelity: ContentOriginValue = "reconstructed"


class Option(BaseModel):
    label: str = Field(description="e.g. A, B, C, D")
    english: str | None = None
    hindi: str | None = None
    visuals: list[Visual] = []


class Statement(BaseModel):
    id: str = Field(description="e.g. I, II, III, IV or A, B, C, D")
    english: str | None = None
    hindi: str | None = None


class MatchingItem(BaseModel):
    id: str
    english: str | None = None
    hindi: str | None = None


class MappingPair(BaseModel):
    key: str
    value: str


class MatchingOptionMapping(BaseModel):
    label: str
    mapping: list[MappingPair] = []


class MatchingData(BaseModel):
    left_column: list[MatchingItem] = []
    right_column: list[MatchingItem] = []
    option_mappings: list[MatchingOptionMapping] = []


class MatrixData(BaseModel):
    rows: list[MatchingItem] = []
    columns: list[MatchingItem] = []
    option_mappings: list[MatchingOptionMapping] = []


class TableCell(BaseModel):
    english: str | None = None
    hindi: str | None = None


class TableRow(BaseModel):
    cells: list[TableCell] = []


class Table(BaseModel):
    headers: list[TableCell] = []
    rows: list[TableRow] = []


class Answer(BaseModel):
    status: Literal["determined", "uncertain"] = "determined"
    correct_option: str | None = None
    answer_text: BilingualText | None = None
    mapping: list[MappingPair] = []
    value: str | None = Field(default=None, description="For numerical answers.")
    unit: str | None = None
    method: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None


class Explanation(BaseModel):
    english: str | None = None
    hindi: str | None = None
    steps: list[str] = []


class OptionAnalysis(BaseModel):
    label: str
    is_correct: bool
    explanation: BilingualText | None = None


class Metadata(BaseModel):
    subject: str | None = None
    sub_subject: str | None = None
    topic: str | None = None
    subtopic: str | None = None
    difficulty: Literal["easy", "moderate", "hard"] | None = None
    cognitive_level: Literal["recall", "understanding", "application", "analysis"] | None = None
    question_category: str | None = None
    concepts: list[str] = []
    keywords: list[str] = []


class SyllabusMapping(BaseModel):
    subject: str | None = None
    unit: str | None = None
    chapter: str | None = None
    topic: str | None = None


class SourceReference(BaseModel):
    page: int | None = None
    question_location: str | None = None
    source_question_number: str | None = None


class ContentOrigin(BaseModel):
    question_text: ContentOriginValue = "source"
    options: ContentOriginValue = "source"
    diagram: ContentOriginValue | None = None
    answer: ContentOriginValue = "generated"
    explanation: ContentOriginValue = "generated"
    metadata: ContentOriginValue = "generated"


class Quality(BaseModel):
    ocr_confidence: float | None = None
    structure_confidence: float | None = None
    answer_confidence: float | None = None
    needs_manual_review: bool = False


class Question(BaseModel):
    id: str = Field(description="e.g. q27")
    question_number: int
    question_type: QuestionType
    question_type_display: str

    question_text: BilingualText
    instructions: BilingualText | None = None

    statements: list[Statement] = []
    assertion: BilingualText | None = None
    reason: BilingualText | None = None
    matching_data: MatchingData | None = None
    matrix: MatrixData | None = None
    table: Table | None = None

    visuals: list[Visual] = []

    options: list[Option]

    answer: Answer
    explanation: Explanation
    option_analysis: list[OptionAnalysis] = []

    metadata: Metadata
    syllabus_mapping: SyllabusMapping
    source_reference: SourceReference
    content_origin: ContentOrigin
    quality: Quality

    notes: list[str] = []


class PageExtraction(BaseModel):
    questions: list[Question]


# ============================================================
# PROMPT
# (adapted from the supplied master prompt; the paper-level
#  framing has been narrowed to "this one page" since the
#  worker only ever sees a single rendered page)
# ============================================================

EXTRACTION_PROMPT = r"""
You are an expert exam-paper digitization, OCR, educational-content
structuring, and question-bank generation system.

You are processing ONE rendered page from an RPSC 2nd Grade Science
Previous Year Question (PYQ) paper. The paper is bilingual (English +
Hindi) and may contain diagrams, figures, chemical structures,
mathematical notation, matching tables, assertion-reason questions,
statement-based questions, matrix questions, and other complex layouts.

Extract every question visible on THIS page accurately and return a JSON
object containing a "questions" array, following the rules below.

============================================================
ACCURACY RULE
============================================================
Treat the page image as the source of truth.
- Do not "improve" wording because another phrasing looks more natural.
- Do not silently invent missing text.
- If text is genuinely unreadable, preserve the most probable reading and
  set quality.needs_manual_review = true; do not fabricate.
- Never represent inferred/generated information as original paper text.

============================================================
BILINGUAL HANDLING
============================================================
For every question, option, statement, assertion/reason, table cell, and
matching/matrix item, preserve English and Hindi SEPARATELY. Do not merge
the two languages into one string. If only one language is actually
present on the page for a given element, set the other to null. Do not
translate the source; translation is out of scope for this pass.

============================================================
QUESTION ORDER AND IDENTITY
============================================================
Preserve the exact printed question_number for each question on this
page. Set "id" to "q" + question_number (e.g. "q27"). Do not renumber
based on OCR guesses.

============================================================
QUESTION TYPE DETECTION
============================================================
Classify every question's "question_type" accurately (single_choice,
multiple_choice, assertion_reason, statement_based, statement_correctness,
matching, matrix_matching, column_matching, sequence_ordering, true_false,
multiple_statements, numerical, diagram_based, graph_based, table_based,
image_based, passage_based, case_based, formula_based, fill_in_the_blank,
other). Do not force every question into single_choice. Also set
"question_type_display" to a human-readable label.

============================================================
ASSERTION-REASON / STATEMENTS / MATCHING / MATRIX / TABLES
============================================================
- Assertion-reason: put the assertion in "assertion" and the reason in
  "reason" (each bilingual), never inside question_text.
- Statement-based: put every statement (I, II, III... or A, B, C...) as a
  separate entry in "statements". Never collapse statements into one
  paragraph.
- Matching: populate matching_data.left_column and right_column
  separately, plus option_mappings (label -> list of key/value pairs)
  when the options encode a mapping (e.g. P-2, Q-4, R-1, S-3).
- Matrix matching: populate matrix.rows and matrix.columns the same way.
- Tables: populate "table" with headers/rows structurally. Never collapse
  a table into plain text.

============================================================
DIAGRAMS, FIGURES, GRAPHS, CHEMICAL STRUCTURES
============================================================
Whenever a question OR an individual option contains a meaningful visual
(diagram, graph, circuit, ray diagram, apparatus, biological figure,
chemical/molecular structure, reaction scheme, taxonomy tree, labeled
figure, or a purely visual answer choice), do NOT rely on a raster image
crop. Instead, reconstruct it as a complete, valid, self-contained SVG:
- Starts with "<svg" and ends with "</svg>".
- Uses a sensible viewBox and standard elements (line, path, circle, rect,
  polygon, polyline, text, etc.).
- Preserves labels, arrows, directions, relationships, and relative
  proportions faithfully — a wrong arrow or missing label can change the
  correct answer.
- No external images, external CSS, or unavailable file references.
Attach the visual to "visuals" on the question if it belongs to the
question stem, or inside the specific option's "visuals" list if it
belongs to that option only. Always also fill "description" with a plain
text summary for accessibility/search. For simple inline math (e.g.
v = u + at), plain text/LaTeX-style notation in the text fields is fine —
reserve SVG for genuinely visual structures.

If, and only if, faithful SVG reconstruction is genuinely impossible for
a specific visual, leave the svg field as a minimal placeholder
("<svg viewBox='0 0 10 10'></svg>"), set source_fidelity to "inferred",
and explain the limitation in that question's "notes" array plus set
quality.needs_manual_review = true. This should be rare.

============================================================
ANSWERS AND EXPLANATIONS
============================================================
Determine the correct answer whenever you reliably can:
- MCQ: answer.correct_option = the option label, plus answer.answer_text.
- Matching/matrix: answer.mapping = list of key/value pairs.
- Numerical: answer.value / answer.unit / answer.method.
- If you cannot reliably determine the answer, set answer.status =
  "uncertain" and leave correct_option/answer_text null. Never invent an
  answer just to fill the field.

Provide a concise bilingual "explanation" (english/hindi) with "steps" for
calculations. For assertion-reason questions, explicitly assess in the
explanation whether the assertion is true, whether the reason is true,
and whether the reason correctly explains the assertion. Explanations
must not contradict the chosen answer.

Populate "option_analysis" (is_correct + short bilingual explanation per
option) especially for conceptual, statement, assertion-reason, and
tricky MCQs.

============================================================
METADATA
============================================================
Fill "metadata" (subject/sub_subject/topic/subtopic/difficulty/
cognitive_level/concepts/keywords) conservatively — use null rather than
inventing an overly specific classification. Fill "syllabus_mapping"
using the likely RPSC/Rajasthan science syllabus context; do not claim
exact official wording unless visible on the page or reliably known.

============================================================
SOURCE TRACEABILITY AND CONTENT ORIGIN
============================================================
Set source_reference.source_question_number to the printed question
number as a string (page/question_location may be left null if not
determinable — do not invent page numbers, the caller fills page).
Set content_origin to distinguish source text ("source") from
reconstructed diagrams ("reconstructed") and generated answer/
explanation/metadata ("generated").

============================================================
QUALITY
============================================================
Set quality.ocr_confidence / structure_confidence / answer_confidence as
numbers between 0 and 1, and needs_manual_review = true for unclear
scans, ambiguous diagrams, uncertain option ordering, or uncertain
answers.

============================================================
FINAL CHECK BEFORE RETURNING
============================================================
Verify, in order: question numbering -> bilingual text completeness ->
option order -> special question-type structures (assertion/reason,
statements, matching, matrix, table) -> every visual is valid SVG
(starts with "<svg", ends with "</svg>") and attached at the right level
(question vs specific option) -> answer/explanation consistency ->
metadata -> valid JSON with no question silently omitted from this page.

Return ONLY JSON matching the provided schema — no markdown fences, no
commentary before or after the JSON.
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
    return any(marker in text for marker in permanent_markers)


def call_gemini_with_retry(client, image_bytes: bytes):
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Gemini attempt {attempt}/{MAX_ATTEMPTS}")

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
                    response_schema=PageExtraction,
                ),
            )

        except Exception as exc:
            last_error = exc
            print(f"Gemini attempt {attempt} failed:")
            print(exc)

            if is_permanent_error(exc):
                print("Permanent API error detected; stopping retries.")
                raise

            if attempt == MAX_ATTEMPTS:
                break

            delay = min(
                RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                RETRY_MAX_SECONDS,
            )
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

    raise RuntimeError(
        f"Gemini request failed after {MAX_ATTEMPTS} attempts."
    ) from last_error


# ============================================================
# VALIDATION
# ============================================================

def validate_svg(svg: str, where: str):
    stripped = svg.strip()
    if not stripped.startswith("<svg"):
        raise ValueError(f"{where}: SVG does not start with '<svg'.")
    if not stripped.endswith("</svg>"):
        raise ValueError(f"{where}: SVG does not end with '</svg>'.")


def validate_page(extraction: PageExtraction):
    seen_numbers = set()

    for question in extraction.questions:
        where = f"Question {question.question_number} ({question.id})"

        if question.question_number <= 0:
            raise ValueError(f"Invalid question number: {question.question_number}")

        if question.question_number in seen_numbers:
            raise ValueError(f"Duplicate question number {question.question_number} on this page.")
        seen_numbers.add(question.question_number)

        if not (question.question_text.english or question.question_text.hindi):
            raise ValueError(f"{where}: question_text has no English or Hindi content.")

        if not question.options:
            # Some question types (numerical, fill_in_the_blank) may legitimately
            # have no discrete options — only enforce for choice-style types.
            if question.question_type in (
                "single_choice",
                "multiple_choice",
                "assertion_reason",
                "statement_based",
                "statement_correctness",
                "matching",
                "matrix_matching",
                "column_matching",
                "sequence_ordering",
                "true_false",
                "multiple_statements",
            ):
                raise ValueError(f"{where}: choice-style question has no options.")

        option_labels = [opt.label for opt in question.options]
        if len(option_labels) != len(set(option_labels)):
            raise ValueError(f"{where}: duplicate option labels {option_labels}.")

        if question.answer.correct_option and question.answer.correct_option not in option_labels:
            raise ValueError(
                f"{where}: answer.correct_option '{question.answer.correct_option}' "
                f"not among option labels {option_labels}."
            )

        # Validate every SVG, wherever it appears.
        for visual in question.visuals:
            validate_svg(visual.svg, f"{where} (question visual)")

        for option in question.options:
            for visual in option.visuals:
                validate_svg(visual.svg, f"{where} option {option.label} visual")

        if not (0 <= (question.quality.ocr_confidence or 0) <= 1):
            raise ValueError(f"{where}: ocr_confidence out of range.")
        if not (0 <= (question.quality.structure_confidence or 0) <= 1):
            raise ValueError(f"{where}: structure_confidence out of range.")
        if not (0 <= (question.quality.answer_confidence or 0) <= 1):
            raise ValueError(f"{where}: answer_confidence out of range.")


# ============================================================
# RESPONSE PARSER
# ============================================================

def parse_response(response_text: str) -> PageExtraction:
    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        extraction = PageExtraction.model_validate_json(cleaned)
    except ValidationError as exc:
        raise ValueError(f"Gemini output failed schema validation:\n{exc}") from exc

    validate_page(extraction)
    return extraction


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/extract_api.py <page>")
        sys.exit(1)

    try:
        page_number = int(sys.argv[1])
    except ValueError:
        print("ERROR: Page number must be an integer.")
        sys.exit(1)

    page_path = os.path.join(PAGES_DIR, f"page_{page_number}.jpg")
    output_path = os.path.join(OUTPUT_DIR, f"page_{page_number}.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(page_path):
        print(f"ERROR: Page image does not exist: {page_path}")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    try:
        with open(page_path, "rb") as file:
            image_bytes = file.read()
    except OSError as exc:
        print(f"ERROR reading image: {exc}")
        sys.exit(1)

    if not image_bytes:
        print("ERROR: Page image is empty.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("======================================")
    print(f"PROCESSING PAGE {page_number}")
    print(f"MODEL: {MODEL_NAME}")
    print("======================================")

    try:
        response = call_gemini_with_retry(client, image_bytes)
    except Exception as exc:
        print(f"ERROR: Gemini failed:\n{exc}")
        sys.exit(1)

    try:
        extraction = parse_response(response.text)
    except Exception as exc:
        print(f"ERROR: Output validation failed:\n{exc}")

        raw_path = os.path.join(OUTPUT_DIR, f"page_{page_number}_raw.txt")
        with open(raw_path, "w", encoding="utf-8") as file:
            file.write(response.text if response.text else "<EMPTY>")
        print(f"Raw response saved to {raw_path}")

        sys.exit(1)

    # Fill in the page number as the authoritative source_reference.page
    for question in extraction.questions:
        question.source_reference.page = page_number

    data = {
        "page_number": page_number,
        "questions": [q.model_dump(mode="json") for q in extraction.questions],
    }

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except OSError as exc:
        print(f"ERROR saving output: {exc}")
        sys.exit(1)

    visual_count = sum(1 for q in extraction.questions if q.visuals or any(o.visuals for o in q.options))
    review_count = sum(1 for q in extraction.questions if q.quality.needs_manual_review)

    print()
    print("SUCCESS")
    print(f"Page: {page_number}")
    print(f"Questions: {len(extraction.questions)}")
    print(f"Questions with visuals: {visual_count}")
    print(f"Questions flagged for manual review: {review_count}")
    print(f"Saved: {output_path}")

    time.sleep(FINAL_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
