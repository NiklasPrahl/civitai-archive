import os
import json

from civitai_manager.src.utils.string_utils import calculate_sha256

CONFIG_FILE = os.environ.get('CONFIG_FILE', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.json')))

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
        return None
    
    for file_path in candidate_paths:
        if calculate_sha256(file_path) == stored_hash:
            return os.path.relpath(file_path, models_dir)

    return None

def load_web_config(config_file_path=None):
    print(f"DEBUG: Attempting to load config from: {config_file_path}")
    if config_file_path is None:
        config_file_path = CONFIG_FILE
        print(f"DEBUG: config_file_path was None, defaulting to: {config_file_path}")
    try:
        if os.path.exists(config_file_path):
            print(f"DEBUG: Config file exists at: {config_file_path}")
            with open(config_file_path, 'r') as f:
                config = json.load(f)
                print(f"DEBUG: Successfully loaded config: {config}")
                
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
                if 'user_images_limit' not in config:
                    config['user_images_limit'] = 0
                if 'user_posts_limit' not in config:
                    config['user_posts_limit'] = 0
                if 'images_per_post_limit' not in config:
                    config['images_per_post_limit'] = 0
                
                return config
        else:
            print(f"DEBUG: Config file does NOT exist at: {config_file_path}")
    except Exception as e:
        print(f"ERROR: Error loading config from {config_file_path}: {e}")
    return {}

def save_web_config(config, config_file_path=None):
    print(f"DEBUG: Attempting to save config to: {config_file_path}")
    print(f"DEBUG: Config data to save: {config}")
    if config_file_path is None:
        config_file_path = CONFIG_FILE
        print(f"DEBUG: config_file_path was None, defaulting to: {config_file_path}")
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        with open(config_file_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"DEBUG: Successfully saved config to: {config_file_path}")
        return True
    except Exception as e:
        print(f"ERROR: Error saving config to {config_file_path}: {e}")
        return False