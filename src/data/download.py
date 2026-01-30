import requests
import zipfile
import io
import os
import shutil


from src.config.config import repo_url, repo_dst_dir, audio_dst_dir, paths_to_delete

def download_and_extract(url, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Downloading from {url}")
    response = requests.get(url)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        print(f"Extracting to {dst_dir}")
        z.extractall(dst_dir)
    print("Done extracting.")

def clean_files(repo_dir, paths_to_delete):
    for f in paths_to_delete:
        path = os.path.join(repo_dir, f)
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {path}")

def move_audio_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    print(f"Moving audio files from {src_dir} to {dst_dir}")
    
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        if os.path.isfile(src_file):
            shutil.move(src_file, dst_file)
    print(f"Moved all audio files to {dst_dir}")

def download_clean():
    # Download and extract
    download_and_extract(repo_url, repo_dst_dir)
    
    # The extracted path will be data/ESC-50-master/
    extracted_dir = os.path.join(repo_dst_dir, "ESC-50-master")
    audio_src_dir = os.path.join(extracted_dir, "audio")
    
    # Clean unwanted files
    clean_files(extracted_dir, paths_to_delete)
    
    # Move audio files to data/audio/0
    move_audio_files(audio_src_dir, audio_dst_dir)
    
    # Clean up the extracted directory
    shutil.rmtree(extracted_dir)
    print(f"Cleanup complete. Audio files are in {audio_dst_dir}")

if __name__ == "__main__":
    download_clean()