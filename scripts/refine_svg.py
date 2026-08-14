"""
scripts/refine_svg.py

WORKFLOW 2 — Refinement & Inline SVG Embedding (Tier 2: Gemini 3.7 Flash)

This script is the "Senior Auditor" pass described in the architecture
plan. It runs AFTER Workflow 1 (drive_sync.py -> extract_api.py ->
merge_jsons.py) has already produced:

    output/<RUN_ID>/question_bank.json
    output/<RUN_ID>/images/*.jpg   (includes the new *_full.jpg crops)

and AFTER you (optionally) hand-cropped extra option-level images into
that same images/ folder using this naming convention:

    q{number}_full.jpg              (whole question block)
    q{number}_question.jpg          (just the question-stem visual)
    q{number}_opt1.jpg ... opt5.jpg (per-option visuals)

Nothing is required to be hand-cropped — if you skip that step, the
script simply falls back to whatever image_full / image_hi / image_en
Workflow 1 already produced.

WHAT THIS SCRIPT DOES, PER QUESTION:
    1. Scrubs any leaked option text out of question_hi / question_en.
    2. Cross-checks the marked correct option + rationales, and fixes
       obvious factual/derivation mistakes.
    3. For questions flagged has_visuals=true, inspects the available
       image(s) and either:
         a) writes clean inline <svg>...</svg> code (preferred), or
         b) writes a structured image_gen_prompt as a fallback when the
            visual is a photograph/organic image that can't be
            reasonably vectorized (e.g. cell histology, terrain photos).

USAGE:
    python3 scripts/refine_svg.py <RUN_ID>
    python3 scripts/refine_svg.py <RUN_ID> --batch-size 15 --delay 15

RATE LIMIT SAFETY (Gemini 3.7 Flash tier: 5 RPM / 250k TPM / 20 RPD):
    - Default batch size is 15 questions per request.
    - Default delay between batches is 15 seconds (~3.5 RPM).
    - Each batch is retried independently on failure; a single bad
      batch never corrupts or blocks the other batches.
    - Progress is written to disk after every batch, so an interrupted
      run can simply be re-run (already-audited questions are skipped
      unless --force is passed).
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
    "gemini-3.7-flash",
)

DEFAULT_BATCH_SIZE = 15
DEFAULT_BATCH_DELAY_SECONDS = 15

MAX_ATTEMPTS = 5
RETRY_BASE_SECONDS = 10
RETRY_MAX_SECONDS = 90

# Images are downscaled before upload to keep token usage predictable.
MAX_IMAGE_DIMENSION = 1024

OUTPUT_ROOT = Path("output")


# ============================================================
# SCHEMAS (Tier-2 audited output)
# ============================================================

class RefinedOption(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]

    hi: str = Field(
        description=(
            "Cleaned Hindi option text. Same meaning as the draft, "
            "with any leaked/duplicated fragments removed."
        )
    )
    en: str = Field(
        description=(
            "Cleaned English option text. Same meaning as the draft, "
            "with any leaked/duplicated fragments removed."
        )
    )

    svg: str | None = Field(
        default=None,
        description=(
            "Inline <svg ...>...</svg> code for THIS option's own "
            "diagram, only if an option-level image was supplied for "
            "this label and it is vectorizable. Otherwise null."
        ),
    )

    image_gen_prompt: str | None = Field(
        default=None,
        description=(
            "Structured external image-generation prompt for THIS "
            "option's diagram, only if an option-level image was "
            "supplied for this label and it is NOT vectorizable "
            "(e.g. a photograph / organic specimen). Otherwise null."
        ),
    )


class RefinedRationale(BaseModel):
    label: Literal["1", "2", "3", "4", "5"]
    is_correct: bool
    rationale_hi: str
    rationale_en: str


class RefinedQuestion(BaseModel):
    number: int

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

    svg_hi: str | None = Field(
        default=None,
        description=(
            "Inline <svg ...>...</svg> for the QUESTION-STEM diagram "
            "(Hindi rendering context), only if a question-level image "
            "was supplied and it is vectorizable. Otherwise null."
        ),
    )
    svg_en: str | None = Field(
        default=None,
        description=(
            "Inline <svg ...>...</svg> for the QUESTION-STEM diagram "
            "(English rendering context), only if a question-level "
            "image was supplied and it is vectorizable. Otherwise "
            "null. If the diagram is language-independent, svg_hi and "
            "svg_en may be identical."
        ),
    )

    image_gen_prompt_hi: str | None = None
    image_gen_prompt_en: str | None = None

    audit_notes: str | None = Field(
        default=None,
        description=(
            "Optional short note on what was fixed/changed. Null if "
            "nothing needed correction."
        ),
    )


class RefinedBatch(BaseModel):
    questions: list[RefinedQuestion]


# ============================================================
# PROMPT
# ============================================================

AUDIT_INSTRUCTIONS = r"""
You are the Tier-2 Senior Auditor for an already-extracted RPSC exam
question bank. A faster, cheaper draft model already extracted these
questions from page images. Your job is NOT to re-extract from scratch.

You will receive a batch of already-extracted questions as JSON,
followed by any available reference images for the visual questions
in this batch (each image is clearly labeled with the question number
and which part of the question it belongs to).

Perform exactly four operations on EVERY question in the batch:

============================================================
1. TEXT & BOUNDARY SCRUBBING
============================================================

- Remove any leaked option text (e.g. "(1) Option A", "(2) Option B")
  that was accidentally appended into question_hi or question_en
  during page OCR.
- Statement-based questions (Statement I, Statement II, ...) must keep
  clean formatting with no trailing choices bleeding into the stem.
- Do not change the underlying meaning of the question or options.
  This is cleanup, not rewriting.

============================================================
2. SCIENTIFIC & ANSWER KEY VERIFICATION
============================================================

- Cross-check the marked correct option and every rationale against
  established scientific/historical fact.
- If an explanation has a subtle math or factual mistake from the
  draft pass, correct the step-by-step derivation.
- Exactly ONE option among labels 1-4 must be is_correct = true.
  Option 5 ("Question not attempted") must always be is_correct =
  false.
- Do not invent a different question. Fix the existing one.

============================================================
3. CODE-BASED DIAGRAM ENGINEERING (only for questions with images
   attached below)
============================================================

For each image you were given for a question, decide between two
paths:

PATH A — Vector drawing (STRONGLY PREFERRED):
If the image is a circuit diagram, geometric figure, ray diagram,
chemical apparatus, logic gate, flowchart, bar/line graph, or any
diagram with clean lines/shapes/text, produce direct, lightweight
inline SVG code: <svg viewBox="0 0 W H" ...>...</svg>
  - Use simple shapes, paths, text, and standard SVG elements.
  - Keep the SVG self-contained (no external references, no <script>).
  - Make it legible at small sizes: use clear stroke widths and
    readable font sizes (>= 12).
  - Reproduce the diagram's actual labeled values/letters exactly as
    shown in the image. Do not invent labels that are not visible.

PATH B — Fallback image-generation prompt:
If the diagram is a photograph, organic/biological specimen (cell
histology, insect anatomy cross-section, plant/animal photo), or a
complex geographic/contour map that cannot be cleanly represented as
clean vector shapes, instead write a structured image_gen_prompt
describing exactly what an external image generator should produce:
subject, composition, labels, style (e.g. "scientific textbook
illustration"), and any text that must appear on it.

Put the RIGHT ONE (never both) in the matching field:
  - Question-level image -> svg_hi/svg_en OR image_gen_prompt_hi/en
  - Option-level image for option X -> options[X].svg OR
    options[X].image_gen_prompt

If NO image was supplied for a question or a specific option, leave
all of its svg / image_gen_prompt fields null. Do not hallucinate a
diagram that was not shown to you.

============================================================
4. AUDIT NOTES
============================================================

If you changed anything meaningful (fixed an answer, corrected a
derivation, removed leaked text), briefly say what in audit_notes.
If nothing needed fixing, set audit_notes to null.

============================================================
OUTPUT RULES
============================================================

- Return ALL questions from the input batch, in the same order, using
  the same "number" values.
- Every question must have exactly 5 options (labels 1-5, same order
  as input) and exactly 5 option_rationales (labels 1-5, same order).
- Do not drop or reorder any question or option.
- Return ONLY JSON matching the provided schema.
"""


# ============================================================
# HELPERS
# ============================================================

def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def load_question_bank(run_dir: Path) -> dict:
    bank_path = run_dir / "question_bank.json"

    if not bank_path.exists():
        print(f"ERROR: {bank_path} does not exist.")
        sys.exit(1)

    try:
        with open(bank_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: Could not read {bank_path}: {exc}")
        sys.exit(1)

    if not isinstance(data.get("questions"), list):
        print(f"ERROR: {bank_path} has no 'questions' array.")
        sys.exit(1)

    return data


def save_question_bank(run_dir: Path, data: dict) -> None:
    bank_path = run_dir / "question_bank.json"
    tmp_path = bank_path.with_suffix(".json.tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        os.replace(tmp_path, bank_path)
    except OSError as exc:
        print(f"ERROR: Could not save {bank_path}: {exc}")
        # Do not exit here — caller decides whether this is fatal,
        # since we may be mid-batch and want to keep whatever progress
        # is already safely on disk from the previous successful save.
        raise


# ============================================================
# IMAGE DISCOVERY (naming convention + JSON fallback)
# ============================================================

def discover_question_images(
    images_dir: Path,
    question: dict,
) -> dict:
    """
    Returns a dict describing which images are available for this
    question:
        {
            "full": Path | None,
            "question": Path | None,
            "options": {"1": Path, "2": Path, ...},
        }
    Search order per slot:
        1. Hand-cropped file following the q{number}_<slot>.* convention.
        2. Fields already present in the question JSON
           (image_full / image_hi / image_en).
    """

    number = question.get("number")

    result = {
        "full": None,
        "question": None,
        "options": {},
    }

    if number is None or not images_dir.exists():
        return result

    extensions = (".jpg", ".jpeg", ".png")

    def find(slot_name: str) -> Path | None:
        for ext in extensions:
            candidate = images_dir / f"q{number}_{slot_name}{ext}"
            if candidate.exists():
                return candidate
        return None

    result["full"] = find("full")
    result["question"] = find("question")

    for label in ("1", "2", "3", "4", "5"):
        found = find(f"opt{label}")
        if found:
            result["options"][label] = found

    # Fall back to whatever Workflow 1 / merge already recorded.
    if result["full"] is None and question.get("image_full"):
        candidate = images_dir / Path(question["image_full"]).name
        if candidate.exists():
            result["full"] = candidate

    if result["question"] is None:
        for key in ("image_hi", "image_en"):
            if question.get(key):
                candidate = images_dir / Path(question[key]).name
                if candidate.exists():
                    result["question"] = candidate
                    break

    return result


def load_image_part(path: Path) -> types.Part | None:
    """Downscale (if needed) and load an image as a genai Part."""

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")

            width, height = image.size
            longest = max(width, height)

            if longest > MAX_IMAGE_DIMENSION:
                scale = MAX_IMAGE_DIMENSION / longest
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale)))
                )

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            image_bytes = buffer.getvalue()

    except (OSError, ValueError) as exc:
        print(f"WARNING: Could not load image {path}: {exc}")
        return None

    if not image_bytes:
        return None

    return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")


# ============================================================
# BATCH REQUEST BUILDING
# ============================================================

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


def build_batch_contents(batch: list[dict], images_dir: Path):
    """
    Returns (contents_list, image_load_warnings) for a single API call.
    """

    contents: list = [AUDIT_INSTRUCTIONS]
    warnings: list[str] = []

    slim_questions = [
        {key: q.get(key) for key in SLIM_FIELDS if key in q}
        for q in batch
    ]

    contents.append(
        "BATCH QUESTIONS (JSON):\n"
        + json.dumps(slim_questions, ensure_ascii=False, indent=2)
    )

    any_images = False

    for question in batch:
        number = question.get("number")

        if not question.get("has_visuals"):
            continue

        images = discover_question_images(images_dir, question)

        if images["full"]:
            part = load_image_part(images["full"])
            if part:
                contents.append(f"IMAGE for Q{number} (full question block):")
                contents.append(part)
                any_images = True
            else:
                warnings.append(f"Q{number}: full image failed to load")

        if images["question"] and images["question"] != images["full"]:
            part = load_image_part(images["question"])
            if part:
                contents.append(f"IMAGE for Q{number} (question stem only):")
                contents.append(part)
                any_images = True
            else:
                warnings.append(f"Q{number}: question image failed to load")

        for label, path in sorted(images["options"].items()):
            part = load_image_part(path)
            if part:
                contents.append(f"IMAGE for Q{number} OPTION {label}:")
                contents.append(part)
                any_images = True
            else:
                warnings.append(f"Q{number} opt{label}: image failed to load")

    if not any_images:
        contents.append(
            "NOTE: No reference images were available for this batch. "
            "Leave all svg / image_gen_prompt fields null for every "
            "question, even those with has_visuals = true."
        )

    return contents, warnings


# ============================================================
# RETRY / API CALL
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


def call_gemini_batch(client, contents):
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"  Gemini attempt {attempt}/{MAX_ATTEMPTS}")

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RefinedBatch,
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
        f"Gemini batch request failed after {MAX_ATTEMPTS} attempts."
    ) from last_error


def parse_batch_response(response_text: str) -> RefinedBatch:
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
        return RefinedBatch.model_validate_json(cleaned)
    except ValidationError as exc:
        raise ValueError(
            f"Gemini batch output failed schema validation:\n{exc}"
        ) from exc


# ============================================================
# MERGE RESULTS BACK INTO THE FULL QUESTION BANK
# ============================================================

def merge_refined_question(original: dict, refined: RefinedQuestion) -> None:
    """Mutates `original` in place. Additive only — nothing outside the
    audited fields is touched (metadata, image paths, source_page,
    has_visuals, visual_location, etc. are all preserved as-is)."""

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

    original["audited"] = True
    original["audit_model"] = MODEL_NAME
    if refined.audit_notes:
        original["audit_notes"] = refined.audit_notes


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Workflow 2: Gemini 3.7 Flash audit + SVG embedding."
    )
    parser.add_argument("run_id", help="Run folder name under output/")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(get_env("REFINE_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=int(
            get_env("REFINE_BATCH_DELAY", str(DEFAULT_BATCH_DELAY_SECONDS))
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-audit questions even if they were already audited.",
    )
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.run_id
    images_dir = run_dir / "images"

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    data = load_question_bank(run_dir)
    all_questions = data["questions"]

    pending = [
        q for q in all_questions
        if args.force or not q.get("audited")
    ]

    print("======================================")
    print("WORKFLOW 2: REFINEMENT & SVG EMBEDDING")
    print("======================================")
    print(f"Run: {args.run_id}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total questions: {len(all_questions)}")
    print(f"Pending audit: {len(pending)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batch delay: {args.delay}s")
    print("======================================")

    if not pending:
        print("Nothing to do — all questions already audited.")
        print("(use --force to re-audit)")
        return

    client = genai.Client(api_key=api_key)

    questions_by_number = {q.get("number"): q for q in all_questions}

    batches = [
        pending[i:i + args.batch_size]
        for i in range(0, len(pending), args.batch_size)
    ]

    total_batches = len(batches)
    failed_batches = 0

    for batch_index, batch in enumerate(batches, start=1):
        numbers = [q.get("number") for q in batch]
        print()
        print(f"--- Batch {batch_index}/{total_batches} "
              f"(questions {numbers}) ---")

        contents, warnings = build_batch_contents(batch, images_dir)
        for warning in warnings:
            print(f"  WARNING: {warning}")

        response = None

        try:
            response = call_gemini_batch(client, contents)
            refined_batch = parse_batch_response(response.text)

        except Exception as exc:
            failed_batches += 1
            print(f"  ERROR: Batch {batch_index} failed permanently: {exc}")

            # If we got a response but it failed schema validation,
            # save the raw text so it can be inspected/debugged later.
            if response is not None and getattr(response, "text", None):
                raw_path = run_dir / f"refine_batch_{batch_index}_raw.txt"
                try:
                    with open(raw_path, "w", encoding="utf-8") as file:
                        file.write(response.text)
                    print(f"  Raw response saved to {raw_path}")
                except OSError as write_exc:
                    print(f"  Could not save raw response: {write_exc}")

            print(f"  Skipping batch {batch_index}; other batches will "
                  f"still be attempted.")
            continue

        refined_by_number = {q.number: q for q in refined_batch.questions}

        merged_count = 0
        for question in batch:
            number = question.get("number")
            refined = refined_by_number.get(number)
            if refined is None:
                print(f"  WARNING: Q{number} missing from Gemini response; "
                      f"leaving it unaudited for this run.")
                continue

            original = questions_by_number.get(number)
            if original is None:
                continue

            merge_refined_question(original, refined)
            merged_count += 1

        print(f"  Merged {merged_count}/{len(batch)} questions.")

        # Fault tolerance: save progress after every batch so a crash
        # or a later failed batch never loses earlier work.
        data["questions"] = list(questions_by_number.values())
        save_question_bank(run_dir, data)
        print(f"  Progress saved to {run_dir / 'question_bank.json'}")

        if batch_index < total_batches:
            print(f"  Waiting {args.delay}s before next batch...")
            time.sleep(args.delay)

    print()
    print("======================================")
    if failed_batches == 0:
        print("WORKFLOW 2 COMPLETE — all batches succeeded")
    else:
        print(f"WORKFLOW 2 FINISHED WITH {failed_batches} FAILED BATCH(ES)")
        print("Re-run this script to retry only the still-unaudited "
              "questions.")
    print("======================================")


if __name__ == "__main__":
    main()
