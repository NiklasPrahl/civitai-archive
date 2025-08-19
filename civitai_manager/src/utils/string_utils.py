import re
import hashlib
import logging
import os # Import os module

def sanitize_filename(filename):
    """
    Sanitize a filename to be safe for various operating systems.
    
    Args:
        filename (str): The original filename.
    Returns:
        str: Sanitized filename
    """
    # Separate base name and extension using os.path.splitext
    base_name, extension = os.path.splitext(filename)
    
    # Replace any character that is not a letter, number, underscore, hyphen, or dot with an underscore
    # This handles special characters and non-ASCII characters
    base_name = re.sub(r'[^a-zA-Z0-9._-]', '_', base_name)
    
    # Collapse multiple underscores to a single one
    base_name = re.sub(r'_+', '_', base_name)
    
    # Collapse multiple dots to a single dot
    base_name = re.sub(r'\.{2,}', '.', base_name)
    
    # Remove leading/trailing underscores and dots
    base_name = base_name.strip('._')
    if not base_name:
        base_name = '_'
    
    # Combine base name and original extension
    final_name = base_name + extension
    
    return final_name


def calculate_sha256(file_path, buffer_size=65536):
    """
    Calculate SHA256 hash of a file
    
    Args:
        file_path: Path to the file
        buffer_size: Size of chunks to read
        
    Returns:
        str: Hex digest of SHA256 hash, or None if file not found
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                sha256_hash.update(data)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return None
    except Exception as e:
        logging.error(f"Error calculating hash for {file_path}: {e}")
        return None