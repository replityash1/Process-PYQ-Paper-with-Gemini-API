"""
scripts/merge_refined_pages.py

WORKFLOW 2 — Merge Step

Takes the per-page refined JSON files produced by refine_page.py
(one worker per page, matrixed like Workflow 1) and folds them back
into the run's final output/<run_id>/question_bank.json, exactly the
way Workflow 1's merge_jsons.py builds it the first time.

Reuses merge_jsons.py's normalize/sort/validate/catalog helpers so
both merge steps behave identically.

Inputs:
    output/<run_id>/raw_pages/page_<N>/   (checked out from git —
        source of the crop images and the original, un-audited
        page_<N>.json / page_<N>_source.jpg)
    downloaded_refined/**/page_<N>.json   (this run's refined pages,
        downloaded as GitHub Actions artifacts)

Output:
    output/<run_id>/question_bank.json    (overwritten)
    output/<run_id>/images/               (refreshed from raw_pages)
    output/catalog.json                   (refreshed)

USAGE:
    python3 scripts/merge_refined_pages.py <run_id>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Both scripts live in scripts/, so this import works when invoked as
# `python3 scripts/merge_refined_pages.py` from the repo root.
import merge_jsons


OUTPUT_ROOT = Path("output")
DOWNLOADED_REFINED_ROOT = Path("downloaded_refined")


# ============================================================
# HELPERS
# ============================================================

def extract_page_number(path: Path) -> int:
    return merge_jsons.extract_page_number(path)


def collect_refined_files():
    if not DOWNLOADED_REFINED_ROOT.exists():
        return []

    files = list(DOWNLOADED_REFINED_ROOT.rglob("page_*.json"))
    files = [f for f in files if not f.name.endswith("_raw.json")]

    files.sort(key=lambda p: (extract_page_number(p), str(p)))

    return files


def load_refined_questions(refined_files):
    questions = []

    for file_path in refined_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            print(f"ERROR reading {file_path}: {exc}")
            continue

        page_number = extract_page_number(file_path)

        for question in data.get("questions", []):
            if not isinstance(question, dict):
                continue

            if not question.get("source_page") and page_number != 10**9:
                question["source_page"] = page_number

            question["_source_file"] = str(file_path)
            questions.append(question)

    return questions


def drop_flagged_duplicates(questions):
    """Remove questions the audit pass explicitly flagged as
    duplicates of another question on the same page. This runs
    BEFORE the fingerprint-based dedup in merge_jsons, which still
    catches any cross-page duplicates the page-level audit couldn't
    see (it only ever looks at one page at a time)."""

    kept = []
    dropped = 0

    for question in questions:
        if question.get("is_duplicate"):
            dropped += 1
            print(
                f"Flagged duplicate removed: "
                f"Q{question.get('number')} "
                f"page={question.get('source_page')} "
                f"(duplicate_of={question.get('duplicate_of_number')})"
            )
            continue
        kept.append(question)

    print(f"Flagged duplicates removed: {dropped}")

    return kept


def load_existing_reviewed_questions(run_dir: Path) -> dict:
    """Load any question_bank.json already committed for this run and
    pull out the ones a human has reviewed in review.html.

    Workflow 2 used to rebuild question_bank.json purely from
    raw_pages/ + freshly downloaded refined-page artifacts, with no
    awareness of the committed file it was about to overwrite. That
    meant re-running Workflow 2 on a run_id after someone had already
    reviewed questions (fixed options, built tables, added inline
    images, etc.) silently discarded all of that work.

    Keyed by (source_page, number), same identity merge_refined_pages
    and review.html both already use to match a question across
    saves."""

    bank_path = run_dir / "question_bank.json"

    if not bank_path.exists():
        return {}

    try:
        with open(bank_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        print(f"WARNING: could not read existing {bank_path}: {exc}")
        return {}

    reviewed = {}

    for question in data.get("questions", []):
        if isinstance(question, dict) and question.get("human_reviewed"):
            key = (question.get("source_page"), question.get("number"))
            reviewed[key] = question

    print(f"Existing human-reviewed questions found: {len(reviewed)}")

    return reviewed


def preserve_human_reviewed(questions, reviewed_by_key: dict):
    """Human-reviewed edits always win over a fresh audit pass.

    Any question that's already been through review.html keeps its
    reviewed version untouched instead of being replaced by this
    run's newly refined/audited copy. Reviewed questions that no
    longer appear in this run's refined output at all (e.g. a page
    got skipped this time) are still carried forward rather than
    dropped."""

    if not reviewed_by_key:
        return questions

    result = []
    seen_keys = set()
    restored = 0

    for question in questions:
        key = (question.get("source_page"), question.get("number"))
        if key in reviewed_by_key:
            result.append(reviewed_by_key[key])
            seen_keys.add(key)
            restored += 1
        else:
            result.append(question)

    for key, question in reviewed_by_key.items():
        if key not in seen_keys:
            print(
                f"NOTE: reviewed Q{question.get('number')} "
                f"page={question.get('source_page')} was not present "
                "in this run's refined output — carrying it forward "
                "as-is anyway."
            )
            result.append(question)
            restored += 1

    print(f"Human-reviewed questions preserved as-is: {restored}")

    return result


def copy_images_from_raw_pages(run_dir: Path, destination_dir: Path) -> int:
    """Images live in the committed raw_pages/ bundles, not in the
    refined-JSON artifacts. Refresh output/<run>/images/ from there."""

    destination_dir.mkdir(parents=True, exist_ok=True)

    raw_pages_dir = run_dir / "raw_pages"

    if not raw_pages_dir.exists():
        print(f"WARNING: {raw_pages_dir} does not exist; no images copied.")
        return 0

    count = 0

    for pattern in ("*.jpg", "*.jpeg"):
        for image_path in raw_pages_dir.rglob(pattern):
            # Source page renders themselves aren't part of the
            # public question bank — only question/option crops are.
            if image_path.name.endswith("_source.jpg"):
                continue

            shutil.copy2(image_path, destination_dir / image_path.name)
            count += 1

    return count


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge page-level refined JSON back into the run's question bank."
    )
    parser.add_argument("run_id", help="Run folder name under output/")
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.run_id
    images_dir = run_dir / "images"

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    print("======================================")
    print("MERGING REFINED PAGES")
    print("======================================")
    print(f"Run: {args.run_id}")

    # --------------------------------------------------------
    # Find refined page artifacts
    # --------------------------------------------------------

    refined_files = collect_refined_files()

    if not refined_files:
        print("ERROR: No refined page JSON artifacts found.")
        sys.exit(1)

    print(f"Refined page files: {len(refined_files)}")

    # --------------------------------------------------------
    # Load, drop flagged dupes, fingerprint-dedup, sort
    # --------------------------------------------------------

    raw_questions = load_refined_questions(refined_files)
    print(f"Raw questions: {len(raw_questions)}")

    raw_questions = drop_flagged_duplicates(raw_questions)

    all_questions = merge_jsons.deduplicate_questions(raw_questions)

    # --------------------------------------------------------
    # Protect human review work from being clobbered by a rerun
    # --------------------------------------------------------

    existing_reviewed = load_existing_reviewed_questions(run_dir)
    all_questions = preserve_human_reviewed(all_questions, existing_reviewed)

    all_questions = merge_jsons.sort_questions(all_questions)

    # --------------------------------------------------------
    # Normalize image references + validate
    # --------------------------------------------------------

    merge_jsons.normalize_image_references(all_questions)
    merge_jsons.validate_questions(all_questions)

    # --------------------------------------------------------
    # Refresh images/ from the committed raw_pages/ bundles
    # --------------------------------------------------------

    image_count = copy_images_from_raw_pages(run_dir, images_dir)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    bank_path = run_dir / "question_bank.json"

    with open(bank_path, "w", encoding="utf-8") as file:
        json.dump(
            {"questions": all_questions},
            file,
            ensure_ascii=False,
            indent=4,
        )

    # --------------------------------------------------------
    # Refresh catalog
    # --------------------------------------------------------

    merge_jsons.write_catalog()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    visual_count = sum(
        1 for q in all_questions if q.get("has_visuals", False)
    )
    audited_count = sum(
        1 for q in all_questions if q.get("audited", False)
    )

    print()
    print("======================================")
    print("MERGE SUCCESS")
    print("======================================")
    print(f"Run: {args.run_id}")
    print(f"Questions: {len(all_questions)}")
    print(f"Visual questions: {visual_count}")
    print(f"Audited: {audited_count}/{len(all_questions)}")
    print(f"Images: {image_count}")
    print(f"Question bank: {bank_path}")


if __name__ == "__main__":
    main()
