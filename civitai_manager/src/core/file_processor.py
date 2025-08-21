import os
from pathlib import Path
import json
import sys
import shutil
from datetime import datetime
import time
import random
import logging
import hashlib
from typing import Optional, Dict, List, Tuple

import requests

from ..utils.string_utils import sanitize_filename, calculate_sha256
from ..utils.html_generators.model_page import generate_html_summary
from ..utils.config import SUPPORTED_FILE_EXTENSIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_export_directories(base_path, safetensors_path):
    """
    Set up the export directories for a model
    
    Args:
        base_path (Path): Base output directory
        safetensors_path (Path): Path to the safetensors file
        
    Returns:
        Path: Model-specific output directory
    """
    sanitized_name = sanitize_filename(safetensors_path.stem)
    model_dir = base_path / sanitized_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def extract_metadata(file_path, output_dir):
    """
    Extract metadata from a file.
    Currently optimized for .safetensors files.
    
    Args:
        file_path (str): Path to the file
        output_dir (Path): Directory to save the output
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        
        base_name = sanitize_filename(path.stem)
        metadata_path = output_dir / f"{base_name}_metadata.json"

        ext = path.suffix.lower()
        if ext == '.safetensors':
            # Logic for .safetensors files
            with open(path, 'rb') as f:
                header_length = int.from_bytes(f.read(8), 'little')
                header_bytes = f.read(header_length)
                header_str = header_bytes.decode('utf-8')
                
                try:
                    header_data = json.loads(header_str)
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        if "__metadata__" in header_data:
                            json.dump(header_data["__metadata__"], f, indent=4)
                        else:
                            json.dump(header_data, f, indent=4)
                    print(f"Metadata successfully extracted to {metadata_path}")
                    return True
                
                except json.JSONDecodeError:
                    print("Error: Could not parse metadata JSON from .safetensors file")
                    return False
        elif ext in {'.ckpt', '.pt', '.pth', '.bin'}:
            # Minimal metadata for non-safetensors formats
            try:
                stat = path.stat()
                meta = {
                    "file_name": path.name,
                    "file_extension": ext,
                    "file_size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "note": "No embedded metadata extracted for this format; using basic file info."
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=4)
                print(f"Basic metadata saved to {metadata_path}")
                return True
            except Exception as e:
                print(f"Error writing basic metadata: {e}")
                return False
        else:
            # Unsupported for metadata
            print(f"Warning: Metadata extraction for {path.suffix} files is not supported.")
            return False
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def extract_hash(file_path, output_dir):
    """
    Calculate hash of a .safetensors file and save it as JSON
    
    Args:
        file_path (str): Path to the .safetensors file
        output_dir (Path): Directory to save the output
    Returns:
        str: Hash value if successful, None otherwise
    """
    try:
        path = Path(file_path)
        output_dir = Path(output_dir)
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        
        hash_value = calculate_sha256(path)
        if not hash_value:
            raise ValueError("Failed to calculate hash")
            
        base_name = sanitize_filename(path.stem)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        hash_path = output_dir / f"{base_name}_hash.json"
        hash_data = {
            "hash_type": "SHA256",
            "hash_value": hash_value,
            "filename": path.name,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(hash_path, 'w', encoding='utf-8') as f:
            json.dump(hash_data, f, indent=4)
        print(f"Hash successfully saved to {hash_path}")
        
        return hash_value
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def download_preview_image(image_url, output_dir, base_name, index=None, is_video=False, image_data=None, subdir: Optional[str] = None):
    """
    Download a preview image from Civitai
    
    Args:
        image_url (str): URL of the image to download
        output_dir (Path): Directory to save the image
        base_name (str): Base name of the safetensors file
        index (int, optional): Image index for multiple images
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not image_url:
            return False
            
        full_size_url = image_url
        
        print("\nDownloading preview image:")
        print(f"URL: {full_size_url}")
        
        # Ensure subdirectory exists if provided
        target_dir = Path(output_dir)
        if subdir:
            target_dir = target_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        response = requests.get(full_size_url, stream=True, timeout=10)
        if response.status_code == 200:
            ext = '.mp4' if is_video else Path(full_size_url).suffix
            sanitized_base = sanitize_filename(base_name)
            image_filename = f'{sanitized_base}_preview{f"_{index}" if index is not None else ""}{ext}'
            image_path = target_dir / image_filename
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Download and save the metadata associated with the image
            if image_data:
                json_filename = f"{Path(image_filename).stem}.json"
                json_path = target_dir / json_filename
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(image_data, f, indent=4)

            print(f"Preview image successfully saved to {image_path}")
            # Return path relative to output_dir for web serving
            rel_path = image_path.relative_to(output_dir)
            return str(rel_path)
        else:
            print(f"Error: Could not download image (Status code: {response.status_code})")
            return None
            
    except Exception as e:
        print(f"Error downloading preview image: {str(e)}")
        return None

def update_missing_files_list(base_path, safetensors_path, status_code):
    """
    Update the list of files missing from Civitai
    
    Args:
        base_path (Path): Base output directory path
        safetensors_path (Path): Path to the safetensors file
        status_code (int): HTTP status code from Civitai API
    """
    missing_file = base_path / "missing_from_civitai.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entries = []
    if missing_file.exists():
        with open(missing_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    filename = line.split(' | ')[-1]
                    if filename != safetensors_path.name:
                        entries.append(line)
    
    if status_code is not None:
        new_entry = f"{timestamp} | Status {status_code} | {safetensors_path.name}"
        entries.append(new_entry)
    
    if entries:
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write("# Files not found on Civitai\n")
            f.write("# Format: Timestamp | Status Code | Filename\n")
            f.write("# This file is automatically updated when the script runs\n")
            f.write("# A file is removed from this list when it becomes available again\n\n")
            
            for entry in sorted(entries, reverse=True):
                f.write(f"{entry}\n")
    elif missing_file.exists():
        missing_file.unlink()
        print("\nAll models are now available on Civitai. Removed missing_from_civitai.txt")

def fetch_version_data(
    hash_value: str,
    output_dir: Path,
    base_path: Path,
    safetensors_path: Path,
    download_all_images: bool = False,
    skip_images: bool = False,
    session: Optional[requests.Session] = None
) -> Optional[int]:
    """
    Fetch version data from Civitai API using file hash
    
    Args:
        hash_value: SHA256 hash of the file
        output_dir: Directory to save the output
        base_path: Base output directory path
        safetensors_path: Path to the safetensors file
        download_all_images: Whether to download all available preview images
        skip_images: Whether to skip downloading images completely
        session: Optional requests session for HTTP calls
        
    Returns:
        Optional[int]: modelId if successful, None otherwise
    """
    local_preview_image_filename = None
    try:
        civitai_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
        print("\nFetching version data from Civitai API:")
        print(civitai_url)
        
        session = session or requests.Session()
        response = session.get(civitai_url)

        base_name = sanitize_filename(safetensors_path.stem)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        civitai_path = output_dir / f"{base_name}_civitai_model_version.json"
        
        if response.status_code == 200:
            response_data_to_save = response.json()
            if local_preview_image_filename:
                response_data_to_save['local_preview_image'] = local_preview_image_filename
            with open(civitai_path, 'w', encoding='utf-8') as f:
                json.dump(response_data_to_save, f, indent=4)
                print(f"Version data successfully saved to {civitai_path}")

                if not skip_images and 'images' in response_data_to_save and response_data_to_save['images']:
                    print(f"\nDownloading all preview images ({len(response_data_to_save['images'])} images found)")
                    previews_subdir = 'previews'
                    for i, image_data in enumerate(response_data_to_save['images']):
                        if 'url' in image_data:
                            is_video = image_data.get('type') == 'video'
                            downloaded_filename = download_preview_image(
                                image_data['url'],
                                output_dir,
                                base_name,
                                i,
                                is_video,
                                image_data,
                                subdir=previews_subdir
                            )
                            if downloaded_filename and not local_preview_image_filename:
                                local_preview_image_filename = downloaded_filename
                            if i < len(response_data_to_save['images']) - 1:
                                time.sleep(1)

                return response_data_to_save.get('modelId')
        else:
            error_message = {
                "error": "Failed to fetch Civitai data",
                "status_code": response.status_code,
                "timestamp": datetime.now().isoformat()
            }
            with open(civitai_path, 'w', encoding='utf-8') as f:
                json.dump(error_message, f, indent=4)
            print(f"Error: Failed to fetch Civitai data (Status code: {response.status_code})")
            
            update_missing_files_list(base_path, safetensors_path, response.status_code)
            return None
                
    except Exception as e:
        print(f"Error fetching version data: {str(e)}")
        return None

def fetch_user_images(
    model_id: int,
    output_dir: Path,
    base_name: str,
    limit: int = 0,
    session: Optional[requests.Session] = None,
    model_version_id: Optional[int] = None,
) -> int:
    """
    Fetch user-generated images for a model and save them under user_images subfolder.

    Args:
        model_id: Civitai model ID
        output_dir: Model output directory
        base_name: Base model name (sanitized)
        limit: Number of images to fetch (0 disables)
        session: Optional requests session

    Returns:
        int: Number of images downloaded
    """
    if not limit or limit <= 0:
        return 0

    try:
        session = session or requests.Session()
        # Use browser-like headers to reduce likelihood of being blocked by edge protection
        session.headers.update({
            'Accept': 'application/json, text/plain, */*',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
            'Referer': 'https://civitai.com/'
        })
        # Prefer modelVersionId when available to ensure exact match to the processed model version
        params = {
            ('modelVersionId' if model_version_id else 'modelId'): (model_version_id if model_version_id else model_id),
            'limit': limit
        }
        url = 'https://civitai.com/api/v1/images'
        print(f"\nFetching user images: {url} params={params}")
        # Retry a few times to tolerate maintenance/WAF hiccups
        data = None
        last_status = None
        for attempt in range(3):
            resp = session.get(url, params=params, timeout=20)
            last_status = resp.status_code
            if resp.status_code != 200:
                print(f"Error fetching user images (Status code: {resp.status_code}) on attempt {attempt+1}")
                time.sleep(1.5 * (attempt + 1))
                continue

            ctype = resp.headers.get('Content-Type', '')
            if 'application/json' not in ctype:
                text_head = (resp.text or '')[:160].replace('\n', ' ')
                print(f"Warning: Expected JSON but got Content-Type '{ctype}' (attempt {attempt+1}). Body: {text_head}")
                time.sleep(1.5 * (attempt + 1))
                continue

            try:
                data = resp.json()
                break
            except Exception as je:
                print(f"Warning: Failed to parse user images JSON on attempt {attempt+1}: {je}")
                time.sleep(1.5 * (attempt + 1))
                continue

        if data is None:
            print(f"Giving up fetching user images after retries. Last status: {last_status}")
            return 0
        # API may return {'items': [...], 'metadata': {...}} or a list directly
        items = data.get('items') if isinstance(data, dict) else data
        # Safety filter in case API ignored/changed filtering: keep only exact modelVersionId when provided
        if model_version_id and items:
            try:
                items = [it for it in items if it.get('modelVersionId') == model_version_id]
            except Exception:
                pass
        if (not items) and model_version_id:
            # Fallback: try with modelId if version-specific query returns nothing
            try:
                fb_params = {'modelId': model_id, 'limit': limit}
                print(f"No items for modelVersionId={model_version_id}. Falling back to modelId with params={fb_params}")
                fb_resp = session.get(url, params=fb_params, timeout=20)
                if fb_resp.status_code == 200 and 'application/json' in fb_resp.headers.get('Content-Type', ''):
                    data = fb_resp.json()
                    items = data.get('items') if isinstance(data, dict) else data
                    if model_version_id and items:
                        # Keep only those that actually reference this version id
                        try:
                            items = [it for it in items if it.get('modelVersionId') == model_version_id]
                        except Exception:
                            pass
                else:
                    print(f"Fallback request failed: status={fb_resp.status_code}, ctype={fb_resp.headers.get('Content-Type')}")
            except Exception as fe:
                print(f"Warning: Fallback to modelId failed: {fe}")

        if not items:
            return 0

        user_subdir = Path(output_dir) / 'user_images'
        user_subdir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for i, item in enumerate(items):
            try:
                image_url = item.get('url') or item.get('meta', {}).get('url')
                if not image_url:
                    continue

                # Build filename
                ext = Path(image_url.split('?')[0]).suffix or '.jpg'
                # Use a simple user image naming to avoid collisions
                filename = f"{sanitize_filename(base_name)}_user_{i}{ext}"
                target_path = user_subdir / filename

                # Download image
                r = session.get(image_url, stream=True, timeout=20)
                if r.status_code == 200:
                    with open(target_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Save metadata next to image
                    meta = item
                    json_path = target_path.with_suffix('.json')
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump(meta, jf, indent=4)

                    downloaded += 1
                else:
                    print(f"Failed to download user image (status {r.status_code})")
            except Exception as ie:
                print(f"Error downloading user image: {ie}")
                continue

        print(f"Downloaded {downloaded} user images")
        return downloaded
    except Exception as e:
        print(f"Error in fetch_user_images: {e}")
        return 0

def fetch_model_details(
    model_id: int,
    output_dir: Path,
    safetensors_path: Path,
    session: Optional[requests.Session] = None
) -> bool:
    """
    Fetch detailed model information from Civitai API
    
    Args:
        model_id: The model ID from Civitai
        output_dir: Directory to save the output
        safetensors_path: Path to the safetensors file
        session: Optional requests session for HTTP calls
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        civitai_model_url = f"https://civitai.com/api/v1/models/{model_id}"
        print("\nFetching model details from Civitai API:")
        print(civitai_model_url)
        
        session = session or requests.Session()
        response = session.get(civitai_model_url)
        
        base_name = sanitize_filename(safetensors_path.stem)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_data_path = output_dir / f"{base_name}_civitai_model.json"
        
        with open(model_data_path, 'w', encoding='utf-8') as f:
            if response.status_code == 200:
                json.dump(response.json(), f, indent=4)
                print(f"Model details successfully saved to {model_data_path}")
                return True
            else:
                error_data = {
                    "error": "Failed to fetch model details",
                    "status_code": response.status_code,
                    "timestamp": datetime.now().isoformat()
                }
                json.dump(error_data, f, indent=4)
                print(f"Error: Could not fetch model details (Status code: {response.status_code})")
                return False
                
    except Exception as e:
        print(f"Error fetching model details: {str(e)}")
        return False

def check_for_updates(safetensors_path, output_dir, hash_value):
    """
    Check if the model needs to be updated by comparing updatedAt timestamps
    
    Args:
        safetensors_path (Path): Path to the safetensors file
        output_dir (Path): Directory where files are saved
        hash_value (str): SHA256 hash of the safetensors file
        
    Returns:
        bool: True if update is needed, False if files are up to date
    """
    try:
        # Check if files exist
        civitai_version_file = output_dir / "civitai_version.txt"
        if not civitai_version_file.exists():
            return True
            
        # Read existing version data
        try:
            with open(civitai_version_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_updated_at = existing_data.get('updatedAt')
                if not existing_updated_at:
                    return True
        except (json.JSONDecodeError, KeyError):
            return True
            
        # Fetch current version data from Civitai
        civitai_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
        print("\nChecking for updates from Civitai API:")
        print(civitai_url)
        
        response = requests.get(civitai_url)
        if response.status_code != 200:
            print(f"Error checking for updates (Status code: {response.status_code})")
            return True
            
        current_data = response.json()
        current_updated_at = current_data.get('updatedAt')
        
        if not current_updated_at:
            return True
            
        # Compare timestamps
        if current_updated_at == existing_updated_at:
            print(f"\nModel {safetensors_path.name} is up to date (Last updated: {existing_updated_at})")
            return False
        else:
            print(f"\nUpdate available for {safetensors_path.name}")
            print(f"Current version: {existing_updated_at}")
            print(f"New version: {current_updated_at}")
            return True
            
    except Exception as e:
        print(f"Error checking for updates: {str(e)}")
        return True

def process_single_file(
    file_path: Path, # Renamed from safetensors_path to file_path
    base_output_path: Path,
    download_all_images: bool = False,
    skip_images: bool = False,
    html_only: bool = False,
    only_update: bool = False,
    session: Optional[requests.Session] = None,
    user_images_limit: int = 0
) -> bool:
    """
    Process a single file
    
    Args:
        file_path: Path to the file
        base_output_path: Base path for output
        download_all_images: Whether to download all available preview images
        skip_images: Whether to skip downloading images completely
        html_only: Whether to only generate HTML files
        only_update: Whether to only update existing processed files
        session: Optional requests session for HTTP calls
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    if not file_path.exists():
        return False
        
    if file_path.suffix not in SUPPORTED_FILE_EXTENSIONS:
        print(f"Skipping unsupported file type: {file_path.name}")
        return False
    
    model_output_dir = setup_export_directories(base_output_path, file_path)
    
    print(f"\nProcessing: {file_path.name}")
    if not html_only:
        print(f"Files will be saved in: {model_output_dir}")
    
    if html_only:
        # Check if required files exist
        base_name = sanitize_filename(file_path.stem)
        required_files = [
            model_output_dir / f"{base_name}_civitai_model.json",
            model_output_dir / f"{base_name}_civitai_model_version.json",
            model_output_dir / f"{base_name}_hash.json"
        ]
        
        if not all(f.exists() for f in required_files):
            return False
            
        generate_html_summary(model_output_dir, file_path)
        return True
    
    if only_update:
        hash_file = model_output_dir / f"{file_path.stem}_hash.json"
        if not hash_file.exists():
            return False
            
        # Read existing hash
        try:
            with open(hash_file, 'r') as f:
                hash_data = json.load(f)
                hash_value = hash_data.get('hash_value')
                if not hash_value:
                    raise ValueError("Invalid hash file")
        except Exception:
            return False
    else:
        hash_value = extract_hash(file_path, model_output_dir)
        if not hash_value:
            return False
    
    if not check_for_updates(file_path, model_output_dir, hash_value):
        return True
    
    metadata_extracted = False
    if only_update:
        metadata_extracted = True # Assume metadata is already there for update mode
    else:
        metadata_extracted = extract_metadata(file_path, model_output_dir)

    if metadata_extracted:
        model_id = fetch_version_data(hash_value, model_output_dir, base_output_path, 
                                    file_path, download_all_images, skip_images, session=session)
        if model_id:
            fetch_model_details(model_id, model_output_dir, file_path, session=session)
            # Fetch user images if configured
            if not skip_images and user_images_limit:
                try:
                    # Load version JSON to get the exact modelVersionId
                    version_json_path = model_output_dir / f"{sanitize_filename(file_path.stem)}_civitai_model_version.json"
                    model_version_id = None
                    try:
                        if version_json_path.exists():
                            with open(version_json_path, 'r', encoding='utf-8') as vf:
                                vdata = json.load(vf)
                                model_version_id = vdata.get('id')
                    except Exception:
                        model_version_id = None
                    fetch_user_images(
                        model_id,
                        model_output_dir,
                        sanitize_filename(file_path.stem),
                        user_images_limit,
                        session=session,
                        model_version_id=model_version_id,
                    )
                except Exception as e:
                    print(f"Warning: failed to fetch user images: {e}")
            generate_html_summary(model_output_dir, file_path)
            return True
        else:
            return False
            
    return False