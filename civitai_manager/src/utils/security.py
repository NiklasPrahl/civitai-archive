import os
from pathlib import Path
from typing import Optional, Tuple

def validate_path(base_path: str, requested_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validates if a requested path is safe and within the base path.
    
    Args:
        base_path: The root directory that should not be escaped
        requested_path: The path to validate
        
    Returns:
        Tuple of (is_valid, absolute_path)
        - is_valid: True if path is safe, False otherwise
        - absolute_path: The validated absolute path or None if invalid
    """
    try:
        # Convert to absolute paths
        base_abs = os.path.abspath(base_path)
        requested_abs = os.path.abspath(os.path.join(base_path, requested_path))
        
        # Check if the requested path is within base path
        if not requested_abs.startswith(base_abs):
            return False, None
            
        # Check if path exists
        if not os.path.exists(requested_abs):
            return False, None
            
        return True, requested_abs
        
    except Exception:
        return False, None

def check_directory_access(path: str, require_write: bool = False) -> bool:
    """
    Checks if a directory exists and has the required permissions.
    
    Args:
        path: Directory path to check
        require_write: If True, checks for write permissions as well
        
    Returns:
        bool: True if directory is accessible with required permissions
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_dir():
            return False
            
        # Check read access
        if not os.access(path, os.R_OK):
            return False
            
        # Check write access if required
        if require_write and not os.access(path, os.W_OK):
            return False
            
        return True
        
    except Exception:
        return False
