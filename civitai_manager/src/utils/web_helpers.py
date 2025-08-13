import os
import json

from civitai_manager.src.utils.string_utils import calculate_sha256

CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')) # Adjusted path

def find_model_file_path(models_dir, stored_hash, stored_filename):
    """
    Finds a model file recursively and verifies it with a hash check.
    Returns the relative path to the model file or None if not found/verified.
    """
    candidate_paths = []
    for root, _, files in os.walk(models_dir):
        if stored_filename in files:
            candidate_paths.append(os.path.join(root, stored_filename))

    if not candidate_paths:
        return None  # File not found by name

    # Check candidates against the stored hash
    for file_path in candidate_paths:
        if calculate_sha256(file_path) == stored_hash:
            return os.path.relpath(file_path, models_dir)

    return None  # No file found with a matching hash

def load_web_config():
    """Load configuration for web interface"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                
                # Ensure all required fields exist with defaults
                if 'models_directory' not in config:
                    config['models_directory'] = ''
                if 'output_directory' not in config:
                    config['output_directory'] = ''
                if 'download_all_images' not in config:
                    config['download_all_images'] = False
                if 'notimeout' not in config:
                    config['notimeout'] = False
                if 'skip_images' not in config:
                    config['skip_images'] = False
                
                return config
    except Exception as e:
        print(f"DEBUG: Error loading config: {e}")
    return {}

def save_web_config(config):
    """Save configuration for web interface"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False