import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DOWNLOAD_ROOT = Path(
    "downloaded_outputs"
)

OUTPUT_ROOT = Path(
    "output"
)

DRIVE_INFO_PATH = Path(
    ".drive_file_info"
)

DEFAULT_RUN_NAME = (
    "default_run"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def question_fingerprint(
    question: dict,
) -> str:

    parts = [
        normalize_text(
            question.get(
                "question_hi",
                "",
            )
        ),
        normalize_text(
            question.get(
                "question_en",
                "",
            )
        ),
    ]

    for option in question.get(
        "options",
        [],
    ):

        parts.append(
            normalize_text(
                option.get(
                    "hi",
                    "",
                )
            )
        )

        parts.append(
            normalize_text(
                option.get(
                    "en",
                    "",
                )
            )
        )

    combined = "\n".join(
        parts
    )

    return hashlib.sha256(
        combined.encode(
            "utf-8"
        )
    ).hexdigest()


def sanitize_slug(
    value: str,
) -> str:

    value = value.strip()

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    value = value.strip(
        "._-"
    )

    return (
        value[:150]
        or DEFAULT_RUN_NAME
    )


def load_drive_info():
    if not DRIVE_INFO_PATH.exists():
        return {
            "file_id": None,
            "file_name": None,
            "folder_slug": DEFAULT_RUN_NAME,
        }

    lines = (
        DRIVE_INFO_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    )

    file_id = (
        lines[0].strip()
        if len(lines) >= 1
        else None
    )

    file_name = (
        lines[1].strip()
        if len(lines) >= 2
        else None
    )

    if len(lines) >= 3:
        folder_slug = sanitize_slug(
            lines[2]
        )
    elif file_name:
        folder_slug = sanitize_slug(
            Path(
                file_name
            ).stem
        )
    else:
        folder_slug = DEFAULT_RUN_NAME

    return {
        "file_id": file_id,
        "file_name": file_name,
        "folder_slug": folder_slug,
    }


def extract_page_number(
    path: Path,
) -> int:

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


def collect_json_files():

    if not DOWNLOAD_ROOT.exists():
        return []

    files = list(
        DOWNLOAD_ROOT.rglob(
            "*.json"
        )
    )

    files = [
        file
        for file in files
        if not file.name.endswith(
            "_raw.json"
        )
    ]

    files.sort(
        key=lambda path: (
            extract_page_number(
                path
            ),
            str(path),
        )
    )

    return files


def load_questions(
    json_files,
):

    questions = []

    for file_path in json_files:

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(
                    file
                )

        except Exception as exc:

            print(
                f"ERROR reading "
                f"{file_path}: {exc}"
            )

            continue

        questions_data = data.get(
            "questions",
            [],
        )

        if not isinstance(
            questions_data,
            list,
        ):
            print(
                f"WARNING: Invalid questions "
                f"array in {file_path}"
            )
            continue

        page_number = (
            extract_page_number(
                file_path
            )
        )

        for question in (
            questions_data
        ):

            if not isinstance(
                question,
                dict,
            ):
                continue

            if not question.get(
                "source_page"
            ):

                if page_number != 10**9:
                    question[
                        "source_page"
                    ] = page_number

            question[
                "_source_file"
            ] = str(
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

    unique = []

    seen = set()

    for question in questions:

        question_text = (
            normalize_text(
                question.get(
                    "question_hi",
                    "",
                )
            )
            +
            normalize_text(
                question.get(
                    "question_en",
                    "",
                )
            )
        )

        if len(
            question_text
        ) < 10:

            print(
                "WARNING: Skipping "
                "extremely short question."
            )

            continue

        fingerprint = (
            question_fingerprint(
                question
            )
        )

        if fingerprint in seen:

            print(
                "Duplicate removed:"
                f" Q{question.get('number')}"
                f" page={question.get('source_page')}"
            )

            continue

        seen.add(
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

    # Printed exam number is authoritative.
    #
    # source_page is only the secondary tie-breaker.

    return sorted(
        questions,
        key=lambda question: (
            int(
                question.get(
                    "number",
                    10**9,
                )
            ),
            int(
                question.get(
                    "source_page",
                    10**9,
                )
            ),
        ),
    )


# ============================================================
# IMAGE COPYING
# ============================================================

def copy_images(
    destination_dir: Path,
):

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DOWNLOAD_ROOT.exists():
        return 0

    images = []

    for pattern in (
        "*.jpg",
        "*.jpeg",
    ):

        images.extend(
            DOWNLOAD_ROOT.rglob(
                pattern
            )
        )

    copied = 0

    for source in images:

        destination = (
            destination_dir
            / source.name
        )

        try:

            shutil.copy2(
                source,
                destination,
            )

            copied += 1

        except OSError as exc:

            print(
                f"WARNING: Could not copy "
                f"{source}: {exc}"
            )

    return copied


# ============================================================
# IMAGE REFERENCES
# ============================================================

def normalize_image_references(
    questions,
):

    for question in questions:

        if question.get(
            "image_hi"
        ):

            question[
                "image_hi"
            ] = (
                "images/"
                +
                Path(
                    question[
                        "image_hi"
                    ]
                ).name
            )

        if question.get(
            "image_en"
        ):

            question[
                "image_en"
            ] = (
                "images/"
                +
                Path(
                    question[
                        "image_en"
                    ]
                ).name
            )

        question.pop(
            "_source_file",
            None,
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_questions(
    questions,
):

    for question in questions:

        if len(
            question.get(
                "options",
                [],
            )
        ) != 5:

            raise ValueError(
                f"Question "
                f"{question.get('number')} "
                "does not have five options."
            )

        labels = [
            str(
                option.get(
                    "label",
                    "",
                )
            )
            for option in question.get(
                "options",
                [],
            )
        ]

        if labels != [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]:

            raise ValueError(
                f"Question "
                f"{question.get('number')} "
                f"has invalid option labels."
            )


# ============================================================
# CATALOG
# ============================================================

def build_catalog():

    catalog = []

    if not OUTPUT_ROOT.exists():
        return catalog

    for run_dir in (
        OUTPUT_ROOT.iterdir()
    ):

        if not run_dir.is_dir():
            continue

        bank_path = (
            run_dir
            / "question_bank.json"
        )

        if not bank_path.exists():
            continue

        try:

            with open(
                bank_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(
                    file
                )

            questions = data.get(
                "questions",
                [],
            )

            visual_count = sum(
                1
                for question in questions
                if question.get(
                    "has_visuals",
                    False,
                )
            )

            try:

                updated_at = (
                    datetime.fromtimestamp(
                        bank_path.stat().st_mtime,
                        tz=timezone.utc,
                    ).isoformat()
                )

            except OSError:

                updated_at = None

            catalog.append(
                {
                    "id": run_dir.name,

                    "name": run_dir.name,

                    "question_count": len(
                        questions
                    ),

                    "visual_question_count": (
                        visual_count
                    ),

                    "question_bank": (
                        f"output/"
                        f"{run_dir.name}/"
                        f"question_bank.json"
                    ),

                    "updated_at": updated_at,
                }
            )

        except Exception as exc:

            print(
                f"WARNING: Could not add "
                f"{run_dir.name} to catalog: "
                f"{exc}"
            )

    catalog.sort(
        key=lambda item: (
            item.get(
                "updated_at"
            ) or "",
        ),
        reverse=True,
    )

    return catalog


def write_catalog():

    catalog = build_catalog()

    catalog_path = (
        OUTPUT_ROOT
        / "catalog.json"
    )

    with open(
        catalog_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "generated_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "runs": catalog,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Catalog updated: "
        f"{catalog_path}"
    )

    print(
        f"Runs in catalog: "
        f"{len(catalog)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "MERGING RPSC QUESTION BANK"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Run information
    # --------------------------------------------------------

    drive_info = load_drive_info()

    folder_slug = (
        drive_info[
            "folder_slug"
        ]
    )

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

    # --------------------------------------------------------
    # Find artifacts
    # --------------------------------------------------------

    json_files = (
        collect_json_files()
    )

    if not json_files:

        print(
            "ERROR: No JSON artifacts found."
        )

        raise SystemExit(1)

    print(
        f"JSON artifacts: "
        f"{len(json_files)}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    raw_questions = (
        load_questions(
            json_files
        )
    )

    print(
        f"Raw questions: "
        f"{len(raw_questions)}"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    all_questions = (
        deduplicate_questions(
            raw_questions
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    all_questions = (
        sort_questions(
            all_questions
        )
    )

    # --------------------------------------------------------
    # Normalize paths
    # --------------------------------------------------------

    normalize_image_references(
        all_questions
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_questions(
        all_questions
    )

    # --------------------------------------------------------
    # Copy images
    # --------------------------------------------------------

    image_count = copy_images(
        images_dir
    )

    # --------------------------------------------------------
    # Save question bank
    # --------------------------------------------------------

    bank_path = (
        target_dir
        / "question_bank.json"
    )

    with open(
        bank_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "questions": all_questions
            },
            file,
            ensure_ascii=False,
            indent=4,
        )

    # --------------------------------------------------------
    # Update global catalog
    # --------------------------------------------------------

    write_catalog()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    visual_count = sum(
        1
        for question in all_questions
        if question.get(
            "has_visuals",
            False,
        )
    )

    print()
    print(
        "======================================"
    )

    print(
        "MERGE SUCCESS"
    )

    print(
        "======================================"
    )

    print(
        f"Run: {folder_slug}"
    )

    print(
        f"Questions: "
        f"{len(all_questions)}"
    )

    print(
        f"Visual questions: "
        f"{visual_count}"
    )

    print(
        f"Images: "
        f"{image_count}"
    )

    print(
        f"Question bank:"
        f" {bank_path}"
    )


if __name__ == "__main__":
    main()
