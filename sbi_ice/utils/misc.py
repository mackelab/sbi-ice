from pathlib import Path

#Some functions to get useful paths in the project
def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent.parent

def get_output_dir() -> Path:
    return get_project_root() / "out"

def get_data_dir() -> Path:
    return get_project_root() / "data"

def get_data_output_dirs() -> Path:
    return get_data_dir(), get_output_dir()