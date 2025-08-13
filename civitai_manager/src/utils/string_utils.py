import re
import hashlib

def sanitize_filename(filename):
    """
    Create a clean, filesystem-friendly filename
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace brackets, quotes, and special characters with underscores

    sanitized = re.sub(r'[\[\]\(\)\{\}\'"#]', '_', filename)
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
    sanitized = re.sub(r'[^\w\-]', '_', sanitized)
    sanitized = sanitized.strip('._')
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized


def calculate_sha256(file_path, buffer_size=65536):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            sha256_hash.update(data)
    return sha256_hash.hexdigest()
