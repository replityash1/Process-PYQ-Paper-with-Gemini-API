import hashlib
import json
import os
import re
import shutil
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DOWNLOAD_ROOT = Path("downloaded_outputs")
OUTPUT_ROOT = Path("output")

DEFAULT_RUN_NAME = "default_run"


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for exact duplicate detection.

    We deliberately DO NOT aggressively normalize punctuation,
    wording, or partial prefixes because two legitimate exam
    questions can have similar openings.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def question_fingerprint(question: dict) -> str:
    """
    Create a strong deterministic fingerprint from the complete
    bilingual question + all five options.

    This prevents duplicate artifacts from being merged twice,
    while avoiding dangerous fuzzy deduplication.
    """

    parts = [
        normalize_text(
            question.get("question_hi", "")
        ),
        normalize_text(
            question.get("question_en", "")
        ),
    ]

    for option in question.get("options", []):
        parts.append(
            normalize_text(
                option.get("hi", "")
            )
        )
        parts.append(
            normalize_text(
                option.get("en", "")
            )
        )

    combined = "\n".join(parts)

    return hashlib.sha256(
        combined.encode("utf-8")
    ).hexdigest()


def get_run_name() -> str:
    """
    Preserve the existing .drive_file_info convention.

    The third line is treated as the generated run/folder name.
    """

    info_file = Path(".drive_file_info")

    if not info_file.exists():
        return DEFAULT_RUN_NAME

    try:

        lines = info_file.read_text(
            encoding="utf-8"
        ).splitlines()

    except OSError as exc:

        print(
            f"WARNING: Could not read .drive_file_info: {exc}"
        )

        return DEFAULT_RUN_NAME

    if len(lines) < 3:
        return DEFAULT_RUN_NAME

    slug = lines[2].strip()

    if not slug:
        return DEFAULT_RUN_NAME

    # Protect against accidental path traversal or separators.
    slug = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        slug,
    )

    return slug or DEFAULT_RUN_NAME


def extract_page_from_filename(
    path: Path,
) -> int:
    """
    Extract page number from filenames such as:

    page_1.json
    page_23.json
    """

    match = re.search(
        r"page[_-]?(\d+)",
        path.name,
        re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    return 10**9


def safe_int(
    value,
    default=10**9,
):
    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def collect_json_files():
    """
    Recursively collect all per-page question JSON files.

    Ignores raw debug responses.
    """

    if not DOWNLOAD_ROOT.exists():
        return []

    files = []

    for path in DOWNLOAD_ROOT.rglob("*.json"):

        if path.name.endswith(
            "_raw.json"
        ):
            continue

        files.append(path)

    files.sort(
        key=lambda p: (
            extract_page_from_filename(p),
            str(p),
        )
    )

    return files


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions(json_files):
    questions = []

    for file_path in json_files:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

        except Exception as exc:

            print(
                f"ERROR: Could not read {file_path}: {exc}"
            )

            continue

        page_from_filename = extract_page_from_filename(
            file_path
        )

        page_questions = data.get(
            "questions",
            [],
        )

        if not isinstance(
            page_questions,
            list,
        ):
            print(
                f"WARNING: Invalid questions array in {file_path}"
            )
            continue

        for question in page_questions:

            if not isinstance(
                question,
                dict,
            ):
                print(
                    f"WARNING: Skipping malformed question in {file_path}"
                )
                continue

            # Ensure source page exists even if an old
            # extractor generated this JSON.
            if not question.get(
                "source_page"
            ):
                question["source_page"] = (
                    page_from_filename
                    if page_from_filename != 10**9
                    else None
                )

            question["_source_file"] = str(
                file_path
            )

            questions.append(
                question
            )

    return questions


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_questions(
    questions,
):
    """
    Safe deduplication strategy:

    1. Exact full-content fingerprint.
    2. Same source page + same original question number
       + exact text match.

    We intentionally do NOT perform broad fuzzy matching because
    exam questions can legitimately have very similar wording.
    """

    unique = []

    seen_fingerprints = set()

    for question in questions:

        q_hi = normalize_text(
            question.get(
                "question_hi",
                "",
            )
        )

        q_en = normalize_text(
            question.get(
                "question_en",
                "",
            )
        )

        # Skip clearly broken/empty records.
        if len(
            q_hi + q_en
        ) < 10:
            print(
                "WARNING: Skipping extremely short question."
            )
            continue

        fingerprint = question_fingerprint(
            question
        )

        if fingerprint in seen_fingerprints:

            print(
                "Duplicate removed:",
                question.get("number"),
                question.get("source_page"),
            )

            continue

        seen_fingerprints.add(
            fingerprint
        )

        unique.append(
            question
        )

    return unique


# ============================================================
# SORTING
# ============================================================

def sort_questions(
    questions,
):
    """
    Preserve exam question order.

    Primary order:
        original question number

    Secondary fallback:
        source page

    For typical RPSC papers the printed question number is the
    authoritative ordering.
    """

    return sorted(
        questions,
        key=lambda q: (
            safe_int(
                q.get(
                    "number"
                )
            ),
            safe_int(
                q.get(
                    "source_page"
                )
            ),
        ),
    )


# ============================================================
# IMAGE HANDLING
# ============================================================

def copy_images(
    target_images_dir: Path,
):
    """
    Copy all cropped JPG/JPEG images from downloaded artifacts
    into the final images directory.

    Filenames generated by the extractor include page + question
    number, preventing normal collisions.
    """

    target_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = []

    if not DOWNLOAD_ROOT.exists():
        return image_files

    for pattern in (
        "*.jpg",
        "*.jpeg",
    ):
        image_files.extend(
            DOWNLOAD_ROOT.rglob(pattern)
        )

    image_files = sorted(
        image_files
    )

    copied = []

    for source in image_files:

        destination = (
            target_images_dir
            / source.name
        )

        try:

            if destination.exists():

                # If the same image already exists and has the
                # same size, don't copy it again.
                if (
                    destination.stat().st_size
                    == source.stat().st_size
                ):
                    copied.append(
                        destination
                    )
                    continue

            shutil.copy2(
                source,
                destination,
            )

            copied.append(
                destination
            )

        except OSError as exc:

            print(
                f"WARNING: Could not copy image "
                f"{source}: {exc}"
            )

    return copied


# ============================================================
# NORMALIZE IMAGE REFERENCES
# ============================================================

def normalize_image_references(
    questions,
):
    """
    Convert temporary artifact paths into paths relative to
    the final question_bank.json.

    Final format:

        images/page3_q17_hi.jpg
    """

    for question in questions:

        if question.get(
            "image_hi"
        ):
            question["image_hi"] = (
                "images/"
                + Path(
                    question["image_hi"]
                ).name
            )

        if question.get(
            "image_en"
        ):
            question["image_en"] = (
                "images/"
                + Path(
                    question["image_en"]
                ).name
            )

        # Remove internal processing fields.
        question.pop(
            "_source_file",
            None,
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_questions(
    questions,
):
    for question in questions:

        options = question.get(
            "options",
            [],
        )

        if len(options) != 5:

            raise ValueError(
                f"Question {question.get('number')} "
                f"does not have exactly five options."
            )

        labels = [
            str(
                option.get(
                    "label",
                    "",
                )
            )
            for option in options
        ]

        if labels != [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]:

            raise ValueError(
                f"Question {question.get('number')} "
                f"has invalid option labels: {labels}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )
    print(
        "MERGING, DEDUPLICATING & ORGANIZING"
    )
    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Determine output folder
    # --------------------------------------------------------

    folder_slug = get_run_name()

    target_dir = (
        OUTPUT_ROOT
        / folder_slug
    )

    images_dir = (
        target_dir
        / "images"
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Run folder: {folder_slug}"
    )

    # --------------------------------------------------------
    # Locate JSON artifacts
    # --------------------------------------------------------

    json_files = collect_json_files()

    if not json_files:

        print(
            "WARNING: No JSON files found."
        )

        return

    print(
        f"JSON artifacts found: {len(json_files)}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    raw_questions = load_questions(
        json_files
    )

    print(
        f"Raw questions loaded: {len(raw_questions)}"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    all_questions = deduplicate_questions(
        raw_questions
    )

    print(
        f"Questions after deduplication: "
        f"{len(all_questions)}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    all_questions = sort_questions(
        all_questions
    )

    # --------------------------------------------------------
    # Add internal sequence WITHOUT destroying
    # original exam question number
    # --------------------------------------------------------

    for sequence, question in enumerate(
        all_questions,
        start=1,
    ):

        question["sequence"] = sequence

        # Keep source_page as an integer where possible.
        if question.get(
            "source_page"
        ) is not None:

            question["source_page"] = safe_int(
                question["source_page"],
                default=None,
            )

    # --------------------------------------------------------
    # Image references
    # --------------------------------------------------------

    normalize_image_references(
        all_questions
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_final_questions(
        all_questions
    )

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    copied_images = copy_images(
        images_dir
    )

    print(
        f"Images copied: {len(copied_images)}"
    )

    # --------------------------------------------------------
    # Save final question bank
    # --------------------------------------------------------

    json_output_path = (
        target_dir
        / "question_bank.json"
    )

    final_data = {
        "questions": all_questions
    }

    with open(
        json_output_path,
        "w",
        encoding="utf-8",
    ) as out_file:

        json.dump(
            final_data,
            out_file,
            ensure_ascii=False,
            indent=4,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    visual_questions = sum(
        1
        for q in all_questions
        if q.get(
            "has_visuals",
            False,
        )
    )

    print()
    print(
        "======================================"
    )
    print(
        "MERGE SUCCESSFUL"
    )
    print(
        "======================================"
    )

    print(
        f"Run: {folder_slug}"
    )

    print(
        f"Questions: {len(all_questions)}"
    )

    print(
        f"Visual questions: {visual_questions}"
    )

    print(
        f"Images: {len(copied_images)}"
    )

    print(
        f"Question bank: {json_output_path}"
    )


if __name__ == "__main__":
    main()
