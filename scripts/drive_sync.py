import io
import json
import os
import re
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# ============================================================
# CONFIG
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/drive"
]

DRIVE_INBOX_FOLDER = "RPSC_To_Process"
DRIVE_ARCHIVE_FOLDER = "RPSC_Completed"

INPUT_DIR = Path("input")
PDF_PATH = INPUT_DIR / "exam_paper.pdf"

DRIVE_INFO_PATH = Path(".drive_file_info")


# ============================================================
# HELPERS
# ============================================================

def get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def sanitize_slug(filename: str) -> str:
    """
    Convert a PDF filename into a safe folder name.

    Example:
        RPSC_Prelims_2025.pdf
    ->
        RPSC_Prelims_2025
    """

    name = Path(filename).stem.strip()

    # Replace characters that are unsafe or inconvenient in paths.
    name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    )

    # Collapse repeated underscores.
    name = re.sub(
        r"_+",
        "_",
        name,
    )

    name = name.strip("._-")

    if not name:
        return "default_run"

    return name[:150]


def escape_drive_query_value(value: str) -> str:
    """
    Escape a string used inside a Google Drive query.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive_service():
    sa_key_json = get_env("GCP_SA_KEY")

    if not sa_key_json:
        raise RuntimeError(
            "GCP_SA_KEY environment variable is not configured."
        )

    try:
        sa_info = json.loads(sa_key_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GCP_SA_KEY does not contain valid JSON."
        ) from exc

    credentials = (
        service_account.Credentials
        .from_service_account_info(
            sa_info,
            scopes=SCOPES,
        )
    )

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def find_folder_id(
    service,
    folder_name: str,
) -> str:
    folder_name_escaped = (
        escape_drive_query_value(folder_name)
    )

    query = (
        f"name = '{folder_name_escaped}' "
        "and mimeType = "
        "'application/vnd.google-apps.folder' "
        "and trashed = false"
    )

    results = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name)",
            pageSize=20,
        )
        .execute()
    )

    files = results.get(
        "files",
        [],
    )

    if not files:
        raise FileNotFoundError(
            f"Google Drive folder '{folder_name}' "
            "was not found. Make sure the folder exists "
            "and is shared with the service-account email."
        )

    # Exact name lookup should normally yield one folder.
    # Use the first deterministic result if duplicates exist.
    files.sort(
        key=lambda item: item.get(
            "name",
            "",
        )
    )

    return files[0]["id"]


# ============================================================
# DRIVE METADATA
# ============================================================

def save_drive_info(
    file_id: str,
    file_name: str,
    folder_slug: str,
):
    """
    Format:

    line 1 = Google Drive file ID
    line 2 = original PDF filename
    line 3 = generated output folder slug
    """

    with open(
        DRIVE_INFO_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{file_id}\n"
        )

        f.write(
            f"{file_name}\n"
        )

        f.write(
            f"{folder_slug}\n"
        )


def load_drive_info():
    if not DRIVE_INFO_PATH.exists():
        return None

    with open(
        DRIVE_INFO_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        lines = [
            line.strip()
            for line in f.read().splitlines()
        ]

    if len(lines) < 2:
        raise RuntimeError(
            ".drive_file_info is malformed. "
            "Expected at least file ID and filename."
        )

    file_id = lines[0]
    file_name = lines[1]

    if len(lines) >= 3 and lines[2]:
        folder_slug = lines[2]
    else:
        folder_slug = sanitize_slug(
            file_name
        )

    return {
        "file_id": file_id,
        "file_name": file_name,
        "folder_slug": folder_slug,
    }


# ============================================================
# FETCH
# ============================================================

def fetch_latest_pdf():
    service = get_drive_service()

    inbox_name = get_env(
        "DRIVE_INBOX_FOLDER",
        DRIVE_INBOX_FOLDER,
    )

    inbox_id = find_folder_id(
        service,
        inbox_name,
    )

    query = (
        f"'{inbox_id}' in parents "
        "and mimeType = 'application/pdf' "
        "and trashed = false"
    )

    results = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields=(
                "files("
                "id,"
                "name,"
                "modifiedTime,"
                "createdTime,"
                "size"
                ")"
            ),
            orderBy="modifiedTime desc",
            pageSize=20,
        )
        .execute()
    )

    files = results.get(
        "files",
        [],
    )

    if not files:
        print(
            "NO_FILE: No PDF found in "
            f"Google Drive folder '{inbox_name}'."
        )

        # IMPORTANT:
        # No PDF is not an infrastructure error.
        # The scheduled workflow can finish normally.
        return None

    selected = files[0]

    file_id = selected["id"]
    file_name = selected["name"]

    print(
        "Found PDF:"
    )

    print(
        f"  Name: {file_name}"
    )

    print(
        f"  ID: {file_id}"
    )

    print(
        f"  Modified: "
        f"{selected.get('modifiedTime', 'unknown')}"
    )

    folder_slug = sanitize_slug(
        file_name
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = service.files().get_media(
        fileId=file_id,
    )

    with open(
        PDF_PATH,
        "wb",
    ) as fh:

        downloader = MediaIoBaseDownload(
            fh,
            request,
        )

        done = False

        while not done:

            status, done = (
                downloader.next_chunk()
            )

            if status:
                print(
                    f"Download "
                    f"{int(status.progress() * 100)}%"
                )

    if not PDF_PATH.exists():
        raise RuntimeError(
            "Google Drive download completed "
            "but input/exam_paper.pdf was not created."
        )

    pdf_size = PDF_PATH.stat().st_size

    if pdf_size <= 0:
        raise RuntimeError(
            "Downloaded PDF is empty."
        )

    print(
        f"Downloaded PDF size: {pdf_size:,} bytes"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_drive_info(
        file_id=file_id,
        file_name=file_name,
        folder_slug=folder_slug,
    )

    print(
        f"Output folder slug: {folder_slug}"
    )

    return {
        "file_id": file_id,
        "file_name": file_name,
        "folder_slug": folder_slug,
    }


# ============================================================
# ARCHIVE
# ============================================================

def archive_pdf():

    info = load_drive_info()

    if not info:
        print(
            "WARNING: .drive_file_info not found. "
            "Nothing to archive."
        )
        return

    service = get_drive_service()

    archive_name = get_env(
        "DRIVE_ARCHIVE_FOLDER",
        DRIVE_ARCHIVE_FOLDER,
    )

    inbox_name = get_env(
        "DRIVE_INBOX_FOLDER",
        DRIVE_INBOX_FOLDER,
    )

    archive_id = find_folder_id(
        service,
        archive_name,
    )

    inbox_id = find_folder_id(
        service,
        inbox_name,
    )

    file_id = info["file_id"]
    file_name = info["file_name"]

    print(
        f"Archiving '{file_name}'..."
    )

    (
        service.files()
        .update(
            fileId=file_id,
            addParents=archive_id,
            removeParents=inbox_id,
            fields="id,parents",
        )
        .execute()
    )

    print(
        f"Successfully moved "
        f"'{file_name}' to "
        f"'{archive_name}'."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    action = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else "fetch"
    )

    try:

        if action == "fetch":

            result = fetch_latest_pdf()

            if result is None:
                # No PDF is a normal condition.
                sys.exit(0)

        elif action == "archive":

            archive_pdf()

        else:

            print(
                f"ERROR: Unknown action '{action}'. "
                "Use 'fetch' or 'archive'."
            )

            sys.exit(2)

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
