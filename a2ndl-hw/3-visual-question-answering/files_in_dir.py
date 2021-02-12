import os

def get_files_in_directory(path, include_folders=False):
    """Get all filenames in a given directory, optionally include folders as well"""
    return [f for f in os.listdir(path) if include_folders or os.path.isfile(os.path.join(path, f))]