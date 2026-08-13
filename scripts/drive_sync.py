import json
import os
import re
import sys
from pathlib import Path

import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# ============================================================
# CONFIG
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/drive"
]

DEFAULT_INBOX_FOLDER = (
    "RPSC_To_Process"
)

DEFAULT_ARCHIVE_FOLDER = (
    "RPSC_Completed"
)

INPUT_DIR = Path(
    "input"
)

PDF_PATH = (
    INPUT_DIR
    / "exam_paper.pdf"
)

# FIXED: Removed the dot to prevent artifact zipping issues
DRIVE_INFO_PATH = (
    INPUT_DIR
    / "drive_info.txt"
)


# ============================================================
# HELPERS
# ============================================================

def get_env(
    name: str,
    default: str | None = None,
):
    value = os.environ.get(
        name
    )

    if value is None:
        return default

    value = value.strip()

    return value or default


def sanitize_slug(
    filename: str,
):

    name = Path(
        filename
    ).stem.strip()

    name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    name = name.strip(
        "._-"
    )

    return (
        name[:150]
        or "default_run"
    )


def escape_drive_query(
    value: str,
):

    return (
        value
        .replace(
            "\\",
            "\\\\",
        )
        .replace(
            "'",
            "\\'",
        )
    )


# ============================================================
# DRIVE SERVICE
# ============================================================

def get_drive_service():

    key_json = get_env(
        "GCP_SA_KEY"
    )

    if not key_json:
        raise RuntimeError(
            "GCP_SA_KEY is not configured."
        )

    try:

        service_account_info = (
            json.loads(
                key_json
            )
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "GCP_SA_KEY is not valid JSON."
        ) from exc

    credentials = (
        service_account.Credentials
        .from_service_account_info(
            service_account_info,
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
):

    escaped = escape_drive_query(
        folder_name
    )

    query = (
        f"name = '{escaped}' "
        "and mimeType = "
        "'application/vnd.google-apps.folder' "
        "and trashed = false"
    )

    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name)",
            pageSize=20,
        )
        .execute()
    )

    files = response.get(
        "files",
        [],
    )

    if not files:

        raise FileNotFoundError(
            f"Drive folder '{folder_name}' "
            "was not found."
        )

    files.sort(
        key=lambda item: item.get(
            "name",
            "",
        )
    )

    return files[0]["id"]


# ============================================================
# DRIVE INFO
# ============================================================

def save_drive_info(
    file_id: str,
    file_name: str,
    folder_slug: str,
):

    with open(
        DRIVE_INFO_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"{file_id}\n"
        )

        file.write(
            f"{file_name}\n"
        )

        file.write(
            f"{folder_slug}\n"
        )


def load_drive_info():
    
    info_path = DRIVE_INFO_PATH
    if not info_path.exists():
        info_path = Path("drive_info.txt")
        
    if not info_path.exists():
        return None

    with open(
        info_path,
        "r",
        encoding="utf-8",
    ) as file:

        lines = (
            file.read()
            .splitlines()
        )

    if len(lines) < 2:
        raise RuntimeError(
            "drive_info.txt is malformed."
        )

    file_id = lines[0].strip()
    file_name = lines[1].strip()

    if len(lines) >= 3:
        folder_slug = lines[2].strip()
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
        DEFAULT_INBOX_FOLDER,
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

    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields=(
                "files("
                "id,"
                "name,"
                "modifiedTime,"
                "size"
                ")"
            ),
            orderBy="modifiedTime desc",
            pageSize=20,
        )
        .execute()
    )

    files = response.get(
        "files",
        [],
    )

    if not files:

        print(
            "NO_FILE: No PDF found in "
            f"'{inbox_name}'."
        )

        return None

    selected = files[0]

    file_id = selected["id"]
    file_name = selected["name"]

    print(
        f"Found PDF: {file_name}"
    )

    folder_slug = sanitize_slug(
        file_name
    )

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = (
        service.files()
        .get_media(
            fileId=file_id
        )
    )

    with open(
        PDF_PATH,
        "wb",
    ) as file:

        downloader = (
            MediaIoBaseDownload(
                file,
                request,
            )
        )

        done = False

        while not done:

            status, done = (
                downloader.next_chunk()
            )

            if status:

                print(
                    "Download "
                    f"{int(status.progress() * 100)}%"
                )

    if not PDF_PATH.exists():
        raise RuntimeError(
            "Downloaded PDF was not created."
        )

    if PDF_PATH.stat().st_size == 0:
        raise RuntimeError(
            "Downloaded PDF is empty."
        )

    save_drive_info(
        file_id,
        file_name,
        folder_slug,
    )

    print(
        f"Run slug: {folder_slug}"
    )

    return True


# ============================================================
# ARCHIVE
# ============================================================

def archive_pdf():

    info = load_drive_info()

    if not info:
        print(
            "WARNING: No Drive metadata found."
        )
        return

    service = get_drive_service()

    archive_name = get_env(
        "DRIVE_ARCHIVE_FOLDER",
        DEFAULT_ARCHIVE_FOLDER,
    )

    inbox_name = get_env(
        "DRIVE_INBOX_FOLDER",
        DEFAULT_INBOX_FOLDER,
    )

    archive_id = find_folder_id(
        service,
        archive_name,
    )

    inbox_id = find_folder_id(
        service,
        inbox_name,
    )

    print(
        f"Archiving '{info['file_name']}'..."
    )

    (
        service.files()
        .update(
            fileId=info["file_id"],
            addParents=archive_id,
            removeParents=inbox_id,
            fields="id,parents",
        )
        .execute()
    )

    print(
        "Successfully archived PDF."
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
                sys.exit(0)

        elif action == "archive":

            archive_pdf()

        else:

            print(
                "ERROR: action must be "
                "'fetch' or 'archive'."
            )

            sys.exit(2)

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
