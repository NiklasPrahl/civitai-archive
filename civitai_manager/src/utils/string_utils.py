import re

def sanitize_filename(filename):
    """
    Create a clean, filesystem-friendly filename
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace problematic characters
    # 1. Replace brackets, quotes, and special characters with underscores
    sanitized = re.sub(r'[\[\]\(\)\{\}\'"#]', '_', filename)
    # 2. Replace Windows-unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
    # 3. Replace other problematic characters (spaces, dots, etc)
    sanitized = re.sub(r'[^\w\-]', '_', sanitized)
    
    # Remove any leading/trailing underscores or dots
    sanitized = sanitized.strip('._')
    
    # Replace multiple underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    
    return sanitized

import hashlib

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
