import os 


def ensure_directory_exists(*dirs: str):
    for path in dirs:
        os.makedirs(path, exist_ok=True)
        print(f"Directory {path} created successfully.")
    else:
        print(f"Directory {path} already exists.")

