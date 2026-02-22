import os

def ensure_dir(path: str):
    """
    Membuat folder jika belum ada.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Folder dibuat: {path}")
