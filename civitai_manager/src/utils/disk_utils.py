import os
import shutil

def check_disk_space(path, needed_bytes):
    """
    Check if enough disk space is available
    
    Args:
        path: Target path where the file will be stored
        needed_bytes: Required space in bytes
        
    Returns:
        bool: True if enough space is available, False otherwise
    """
    try:
        total, used, free = shutil.disk_usage(path)
        # Leave 5% buffer
        available = free * 0.95
        return available >= needed_bytes
    except Exception:
        return False
