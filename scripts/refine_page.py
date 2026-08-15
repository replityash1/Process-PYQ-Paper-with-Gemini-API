"""
scripts/refine_page.py

WORKFLOW 2 — Page-Level Refinement (Tier 2: Gemini 3.7 Flash)

This replaces the old question-count batching in refine_svg.py with a
page-level audit that mirrors Workflow 1's own architecture: one
worker per PAGE, run through the same matrix pattern as
scripts/extract_api.py.

Each worker reads back exactly what Workflow 1 produced for ONE page
and never touched again:

    output/<run_id>/raw_pages/page_<N>/page_<N>.json     (draft JSON)
    output/<run_id>/raw_pages/page_<N>/page_<N>_source.jpg (source page)
    output/<run_id>/raw_pages/page_<N>/page<N>_q*_*.jpg    (crops)

Unlike the old Tier-2 pass, this sends the ACTUAL SOURCE PAGE IMAGE
back to Gemini, so the audit is a real compare-against-the-PDF check,
not just an internal consistency pass over crops.

Per page, Gemini is asked to:
    1. Re-check every question's text against the source page image
       and fix transcription errors (leaked option text, OCR
       corruption, missing statements/columns, etc.) — same fidelity
       rules as the Tier-1 extraction prompt.
    2. Verify the marked correct option + rationales are right.
    3. Flag (not silently delete) any question on this page that is a
       duplicate of another question on the SAME page.
    4. Produce inline SVG / image_gen_prompt for visual questions,
       using the crop images as reference.
    5. Note (without inventing content) if a visible question number
       on the page seems to be missing from the draft JSON.

USAGE:
    python3 scripts/refine_page.py <run_id> <page>
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.environ.get(
    "REFINE_MODEL_NAME",
    "gemini-3.5-flash-lite",
)

MAX_ATTEMPTS = 6
RETRY_BASE_SECONDS = 10
RETRY_MAX_SECONDS = 90
FINAL_SLEEP_SECONDS = 8

# Reference images are downscaled before upload to keep token usage
# predictable across pages with many visual questions.
MAX_IMAGE_DIMENSION = 1400

OUTPUT_ROOT = Path("output")
REFINED_OUTPUT_DIR = Path("refined_temp")


# ============================================================
# SCHEMAS
# ============================================================

class RefinedOption(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]

    hi: str = Field(
        description=(
            "Corrected Hindi option text, verified against the "
            "source page image. Same meaning as the draft unless the "
            "draft was wrong."
        )
    )
    en: str = Field(
        description=(
            "Corrected English option text, verified against the "
            "source page image."
        )
    )

    svg: str | None = Field(
        default=None,
        description=(
            "Inline <svg>...</svg> for this option's own diagram, "
            "only if an option-level crop was supplied for this "
            "label and it is vectorizable. Otherwise null."
        ),
    )
    image_gen_prompt: str | None = Field(
        default=None,
        description=(
            "Structured external image-generation prompt for this "
            "option's diagram, only if a crop was supplied and it is "
            "a photograph/organic image that cannot be vectorized. "
            "Otherwise null."
        ),
    )


class RefinedRationale(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]
    is_correct: bool
    rationale_hi: str
    rationale_en: str


class RefinedPageQuestion(BaseModel):
    number: int = Field(
        description="Must match the draft question's original number."
    )

    question_hi: str
    question_en: str

    options: list[RefinedOption] = Field(
        description="Exactly five options, labels 1-5, same order."
    )

    option_rationales: list[RefinedRationale] = Field(
        description="Exactly five rationales, labels 1-5, same order."
    )

    explanation_hi: str
    explanation_en: str

    svg_hi: str | None = None
    svg_en: str | None = None
    image_gen_prompt_hi: str | None = None
    image_gen_prompt_en: str | None = None

    is_duplicate: bool = Field(
        default=False,
        description=(
            "True ONLY if this question is a duplicate of another "
            "question on THIS SAME PAGE (same question/options, "
            "just extracted twice). Do not use this for questions "
            "that are merely similar in topic."
        ),
    )
    duplicate_of_number: int | None = Field(
        default=None,
        description=(
            "If is_duplicate is true, the 'number' of the question "
            "this one duplicates. Otherwise null."
        ),
    )

    audit_notes: str | None = Field(
        default=None,
        description=(
            "Brief note on what was changed and why, citing the "
            "source page image. Null if nothing needed correction."
        ),
    )


class RefinedPage(BaseModel):
    page: int

    questions: list[RefinedPageQuestion] = Field(
        description=(
            "Every question from the input, in the same order, "
            "same 'number' values. Never drop or add questions here "
            "— use is_duplicate to flag, use completeness_note to "
            "report a suspected gap."
        )
    )

    completeness_note: str | None = Field(
        default=None,
        description=(
            "Set ONLY if a question number printed on the source "
            "page image appears to be missing from the input list "
            "entirely (e.g. input jumps from Q12 to Q14). Name the "
            "missing number. Do NOT invent its content. Null if the "
            "page's questions look complete."
        ),
    )


# ============================================================
# PROMPT
# ============================================================

AUDIT_INSTRUCTIONS = r"""
You are the Tier-2 Senior Auditor for an already-extracted RPSC exam
question bank. A faster, cheaper draft model already extracted these
questions from this exact page image. Your job is to VERIFY and FIX,
not to re-extract from scratch or invent a new question.

You are given, in order:
  1. These instructions.
  2. The SOURCE PAGE IMAGE — this is ground truth. Every fact,
     number, statement, and option must match what is actually
     printed on this image.
  3. The full draft JSON for every question the Tier-1 model already
     extracted from this page.
  4. Any crop images available for individual questions/options
     (labeled with question number and slot).

============================================================
WHAT TO CHECK, PER QUESTION
============================================================

1. TEXT FIDELITY AGAINST THE SOURCE PAGE IMAGE
   - Compare question_hi / question_en and every option against the
     actual printed text on the page image.
   - Fix leaked option text bled into the question stem.
   - Fix OCR corruption (garbled symbols, wrong script, missing
     Unicode diacritics).
   - For statement-based questions, confirm every statement
     (Statement I/II/III/IV or (A)/(B)/(C)/(D)) is present and
     correctly numbered.
   - For matrix/matching questions, confirm BOTH Column-I/List-I AND
     Column-II/List-II are present in full.
   - All math/science notation stays wrapped in inline LaTeX ($...$),
     never display math ($$...$$) inside options.
   - Do not paraphrase or "improve" wording that is already correct
     — only fix actual mismatches against the source image.
   - If text is genuinely illegible on the source image, keep
     [UNCLEAR] / [ILLEGIBLE] rather than guessing.

2. ANSWER KEY / RATIONALE VERIFICATION
   - Cross-check the marked correct option and every rationale
     against the source image and established fact.
   - Exactly ONE option among labels 1-4 must have is_correct = true.
     Option 5 ("Question not attempted") is always is_correct = false.

3. DUPLICATE DETECTION (SAME PAGE ONLY)
   - If two questions on this page are the same question extracted
     twice under different numbers, set is_duplicate = true and
     duplicate_of_number on the later/lower-quality copy. Keep both
     in your output — do not delete either. The merge step will drop
     the flagged one.

4. VISUAL QUESTIONS — DIAGRAM ENGINEERING (only where a crop image
   was supplied for that question/option)
   PATH A (preferred) — clean inline SVG (<svg viewBox="0 0 W H" ...>)
     for circuit diagrams, geometry/ray diagrams, graphs, logic
     gates, flowcharts, chemical apparatus: reproduce the actual
     labeled values/letters shown, no invented labels.
   PATH B (fallback) — a structured image_gen_prompt for photographs
     or organic/biological images that cannot be cleanly vectorized.
   Put the SVG or the prompt in the matching field only
   (question-level -> svg_hi/svg_en or image_gen_prompt_hi/en;
   option-level -> options[x].svg or options[x].image_gen_prompt).
   If no crop was supplied for a slot, leave its fields null — never
   hallucinate a diagram you were not shown.

5. COMPLETENESS
   - If the source page image visibly shows a question number that
     is entirely absent from the input JSON, set completeness_note
     naming that number. Do not fabricate its content.

============================================================
OUTPUT RULES
============================================================
- Return the same "page" number as given.
- Return ALL input questions, same order, same "number" values.
- Never drop or add a question object — use is_duplicate /
  completeness_note instead.
- Every question keeps exactly 5 options and 5 option_rationales,
  labels 1-5, same order as input.
- Return ONLY JSON matching the provided schema.
"""


# ============================================================
# HELPERS
# ============================================================

def load_page_bundle(raw_dir: Path, page_number: int) -> dict:
    json_path = raw_dir / f"page_{page_number}.json"

    if not json_path.exists():
        print(f"ERROR: {json_path} does not exist.")
        sys.exit(1)

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Could not read {json_path}: {exc}")
        sys.exit(1)

    if not isinstance(data.get("questions"), list):
        print(f"ERROR: {json_path} has no 'questions' array.")
        sys.exit(1)

    return data


def load_image_part(path: Path, max_dimension: int) -> types.Part | None:
    """Downscale (if needed) and load an image as a genai Part."""

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")

            width, height = image.size
            longest = max(width, height)

            if longest > max_dimension:
                scale = max_dimension / longest
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale)))
                )

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
            image_bytes = buffer.getvalue()

    except (OSError, ValueError) as exc:
        print(f"WARNING: Could not load image {path}: {exc}")
        return None

    if not image_bytes:
        return None

    return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")


SLIM_FIELDS = (
    "number",
    "question_hi",
    "question_en",
    "options",
    "option_rationales",
    "explanation_hi",
    "explanation_en",
    "has_visuals",
    "visual_location",
)


def build_contents(
    page_number: int,
    questions: list[dict],
    raw_dir: Path,
):
    contents: list = [AUDIT_INSTRUCTIONS]

    source_path = raw_dir / f"page_{page_number}_source.jpg"
    if source_path.exists():
        part = load_image_part(source_path, MAX_IMAGE_DIMENSION)
        if part:
            contents.append(
                f"SOURCE PAGE IMAGE for page {page_number} "
                "(ground truth — verify every question against this):"
            )
            contents.append(part)
        else:
            print(f"WARNING: source page image failed to load: {source_path}")
    else:
        print(f"WARNING: source page image not found: {source_path}")

    slim_questions = [
        {key: q.get(key) for key in SLIM_FIELDS if key in q}
        for q in questions
    ]

    contents.append(
        f"DRAFT QUESTIONS FOR PAGE {page_number} (JSON):\n"
        + json.dumps(slim_questions, ensure_ascii=False, indent=2)
    )

    any_crop = False

    for question in questions:
        if not question.get("has_visuals"):
            continue

        number = question.get("number")

        for slot_key, label in (
            ("image_full", "full question block"),
            ("image_hi", "Hindi question stem"),
            ("image_en", "English question stem"),
        ):
            filename = question.get(slot_key)
            if not filename:
                continue

            crop_path = raw_dir / Path(filename).name
            if not crop_path.exists():
                continue

            part = load_image_part(crop_path, MAX_IMAGE_DIMENSION)
            if part:
                contents.append(f"CROP for Q{number} ({label}):")
                contents.append(part)
                any_crop = True

    if not any_crop:
        contents.append(
            "NOTE: No question-level crop images were available for "
            "this page. Use the source page image for all visual "
            "verification, and leave svg/image_gen_prompt fields "
            "null for every question."
        )

    return contents


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


def call_gemini_with_retry(client, contents):
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"  Gemini attempt {attempt}/{MAX_ATTEMPTS}")

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RefinedPage,
                ),
            )

        except Exception as exc:
            last_error = exc
            print(f"  Gemini attempt {attempt} failed: {exc}")

            if is_permanent_error(exc):
                print("  Permanent API error detected; stopping retries.")
                raise

            if attempt == MAX_ATTEMPTS:
                break

            delay = min(
                RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
                RETRY_MAX_SECONDS,
            )
            print(f"  Retrying in {delay} seconds...")
            time.sleep(delay)

    raise RuntimeError(
        f"Gemini request failed after {MAX_ATTEMPTS} attempts."
    ) from last_error


def parse_response(response_text: str) -> RefinedPage:
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
        return RefinedPage.model_validate_json(cleaned)
    except ValidationError as exc:
        raise ValueError(
            f"Gemini output failed schema validation:\n{exc}"
        ) from exc


def merge_refined_question(original: dict, refined: RefinedPageQuestion) -> None:
    """Mutates `original` in place. Additive only — metadata,
    source_page, box fields, and existing image_* paths are all
    preserved as-is; only audited content fields change."""

    original["question_hi"] = refined.question_hi
    original["question_en"] = refined.question_en
    original["explanation_hi"] = refined.explanation_hi
    original["explanation_en"] = refined.explanation_en

    refined_options_by_label = {opt.label: opt for opt in refined.options}
    for option in original.get("options", []):
        label = str(option.get("label"))
        refined_opt = refined_options_by_label.get(label)
        if not refined_opt:
            continue
        option["hi"] = refined_opt.hi
        option["en"] = refined_opt.en
        if refined_opt.svg:
            option["svg"] = refined_opt.svg
        if refined_opt.image_gen_prompt:
            option["image_gen_prompt"] = refined_opt.image_gen_prompt

    refined_rationales_by_label = {
        r.label: r for r in refined.option_rationales
    }
    for rationale in original.get("option_rationales", []):
        label = str(rationale.get("label"))
        refined_r = refined_rationales_by_label.get(label)
        if not refined_r:
            continue
        rationale["is_correct"] = refined_r.is_correct
        rationale["rationale_hi"] = refined_r.rationale_hi
        rationale["rationale_en"] = refined_r.rationale_en

    if refined.svg_hi:
        original["svg_hi"] = refined.svg_hi
    if refined.svg_en:
        original["svg_en"] = refined.svg_en
    if refined.image_gen_prompt_hi:
        original["image_gen_prompt_hi"] = refined.image_gen_prompt_hi
    if refined.image_gen_prompt_en:
        original["image_gen_prompt_en"] = refined.image_gen_prompt_en

    original["is_duplicate"] = refined.is_duplicate
    if refined.duplicate_of_number is not None:
        original["duplicate_of_number"] = refined.duplicate_of_number

    original["audited"] = True
    original["audit_model"] = MODEL_NAME
    if refined.audit_notes:
        original["audit_notes"] = refined.audit_notes


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Workflow 2: page-level Gemini audit against the source PDF page."
    )
    parser.add_argument("run_id", help="Run folder name under output/")
    parser.add_argument("page", type=int, help="Page number to refine")
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.run_id
    raw_dir = run_dir / "raw_pages" / f"page_{args.page}"

    if not raw_dir.exists():
        print(f"ERROR: Raw page directory not found: {raw_dir}")
        print("(Run Workflow 1's merge step first — it populates raw_pages/.)")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    REFINED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_page_bundle(raw_dir, args.page)
    questions = data["questions"]

    print("======================================")
    print("WORKFLOW 2: PAGE-LEVEL REFINEMENT")
    print("======================================")
    print(f"Run: {args.run_id}")
    print(f"Page: {args.page}")
    print(f"Model: {MODEL_NAME}")
    print(f"Questions on page: {len(questions)}")
    print("======================================")

    client = genai.Client(api_key=api_key)

    contents = build_contents(args.page, questions, raw_dir)

    response = None
    try:
        response = call_gemini_with_retry(client, contents)
        refined_page = parse_response(response.text)

    except Exception as exc:
        print(f"ERROR: Page {args.page} refinement failed: {exc}")

        if response is not None and getattr(response, "text", None):
            raw_path = REFINED_OUTPUT_DIR / f"page_{args.page}_raw.txt"
            try:
                with open(raw_path, "w", encoding="utf-8") as file:
                    file.write(response.text)
                print(f"Raw response saved to {raw_path}")
            except OSError as write_exc:
                print(f"Could not save raw response: {write_exc}")

        sys.exit(1)

    questions_by_number = {q.get("number"): q for q in questions}
    refined_by_number = {q.number: q for q in refined_page.questions}

    merged_count = 0
    for number, original in questions_by_number.items():
        refined = refined_by_number.get(number)
        if refined is None:
            print(f"WARNING: Q{number} missing from Gemini response; left unaudited.")
            continue

        merge_refined_question(original, refined)
        merged_count += 1

    if refined_page.completeness_note:
        data["completeness_note"] = refined_page.completeness_note
        print(f"NOTE: {refined_page.completeness_note}")

    data["questions"] = list(questions_by_number.values())

    output_path = REFINED_OUTPUT_DIR / f"page_{args.page}.json"

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except OSError as exc:
        print(f"ERROR saving refined output: {exc}")
        sys.exit(1)

    duplicate_count = sum(
        1 for q in data["questions"] if q.get("is_duplicate")
    )

    print()
    print("SUCCESS")
    print(f"Page: {args.page}")
    print(f"Audited: {merged_count}/{len(questions)}")
    print(f"Flagged duplicates: {duplicate_count}")
    print(f"Saved: {output_path}")

    time.sleep(FINAL_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
