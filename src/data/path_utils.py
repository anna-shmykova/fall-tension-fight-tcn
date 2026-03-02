from pathlib import Path

def collect_json_paths(directory_path, extension='*.json'):
    """
    Recursively finds all files with a given extension using pathlib.
    
    Args:
        directory_path (str): The starting directory to search.
        extension (str): The file extension (e.g., '*.txt', '*.jpg').
        
    Returns:
        list: A list of Path objects for the matching files.
    """
    p = Path(directory_path)
    # Use rglob (recursive glob) with the extension pattern
    files = list(p.rglob(extension))
    return files
