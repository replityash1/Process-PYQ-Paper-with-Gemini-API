import os
import sys
import json
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    sa_key_json = os.environ.get('GCP_SA_KEY')
    if not sa_key_json:
        raise ValueError("GCP_SA_KEY environment variable not found.")
    sa_info = json.loads(sa_key_json)
    creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def find_folder_id(service, folder_name):
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    if not files:
        raise FileNotFoundError(f"Google Drive folder '{folder_name}' not found. Make sure you shared it with the service account email.")
    return files[0]['id']

def fetch_latest_pdf():
    service = get_drive_service()
    inbox_name = os.environ.get('DRIVE_INBOX_FOLDER', 'RPSC_To_Process')
    inbox_id = find_folder_id(service, inbox_name)

    query = f"'{inbox_id}' in parents and mimeType = 'application/pdf' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])

    if not files:
        print("No new PDF found in the Google Drive Inbox folder.")
        return None

    file_id = files[0]['id']
    file_name = files[0]['name']
    print(f"Found PDF to process: {file_name} (ID: {file_id})")

    os.makedirs("input", exist_ok=True)
    output_path = os.path.join("input", "exam_paper.pdf")

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(output_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    with open('.drive_file_info', 'w') as f:
        f.write(f"{file_id}\n{file_name}")

    return file_id, file_name

def archive_pdf():
    if not os.path.exists('.drive_file_info'):
        return
    
    with open('.drive_file_info', 'r') as f:
        lines = f.read().splitlines()
        if len(lines) < 2:
            return
        file_id, file_name = lines[0], lines[1]

    service = get_drive_service()
    archive_name = os.environ.get('DRIVE_ARCHIVE_FOLDER', 'RPSC_Completed')
    archive_id = find_folder_id(service, archive_name)
    inbox_name = os.environ.get('DRIVE_INBOX_FOLDER', 'RPSC_To_Process')
    inbox_id = find_folder_id(service, inbox_name)

    service.files().update(
        fileId=file_id,
        addParents=archive_id,
        removeParents=inbox_id,
        fields='id, parents'
    ).execute()
    print(f"Successfully moved '{file_name}' to {archive_name} folder.")

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'fetch'
    if action == 'fetch':
        res = fetch_latest_pdf()
        if not res:
            sys.exit(1)
    elif action == 'archive':
        archive_pdf()
