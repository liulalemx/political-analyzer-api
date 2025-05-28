import os, gdown, zipfile

def download_and_extract_models():
    models_path = "./models"
    if not os.path.exists(models_path):  # Only download if not already present
        print("Downloading models.zip from Google Drive...")
        file_id = os.environ.get("MODELS_FILE_ID")  # Store in .env or Render env vars
        url = f"https://drive.google.com/uc?id={os.getenv('MODELS_FILE_ID')}"
        output = "models.zip"
        gdown.download(url, output, quiet=False)
        print("Extracting models.zip...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(output)
        print("Models extracted.")
    else:
        print("Models already downloaded, skipping...")
    