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
from typing import Optional, Dict, List, Tuple, DefaultDict

import requests

from ..utils.string_utils import sanitize_filename, calculate_sha256
from ..utils.html_generators.model_page import generate_html_summary
from ..utils.config import SUPPORTED_FILE_EXTENSIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_export_directories(base_path: Path, file_path: Path) -> Path:
    """
    Create and return the model-specific output directory.

    Structure:
      <base_path>/<base_name>/
        - previews/
        - user_images/ (legacy)
        - user_posts/
    """
    base_name = sanitize_filename(file_path.stem)
    model_dir = Path(base_path) / base_name
    (model_dir / 'previews').mkdir(parents=True, exist_ok=True)
    (model_dir / 'user_posts').mkdir(parents=True, exist_ok=True)
    return model_dir

def extract_metadata(file_path: Path, output_dir: Path) -> bool:
    """Extract minimal metadata for a model file into <stem>_metadata.json."""
    try:
        if not file_path.exists():
            return False
        if file_path.suffix not in SUPPORTED_FILE_EXTENSIONS:
            return False
        meta = {
            'filename': file_path.name,
            'size_bytes': file_path.stat().st_size,
            'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{file_path.stem}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=4)
        return True
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return False

def extract_hash(file_path: Path, output_dir: Path) -> Optional[str]:
    """Compute SHA-256 for the file and write <stem>_hash.json."""
    try:
        if not file_path.exists():
            return None
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                if not chunk:
                    break
                h.update(chunk)
        value = h.hexdigest()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{file_path.stem}_hash.json", 'w', encoding='utf-8') as f:
            json.dump({'hash_value': value, 'algorithm': 'SHA256'}, f, indent=4)
        return value
    except Exception as e:
        print(f"Error extracting hash: {e}")
        return None

def fetch_user_posts(
    model_id: int,
    output_dir: Path,
    base_name: str,
    posts_limit: int = 0,
    images_per_post_limit: int = 0,
    session: Optional[requests.Session] = None,
    model_version_id: Optional[int] = None,
    user_images_level: str = 'ALL',
) -> Dict[int, int]:
    """
    Fetch user posts (grouped by postId) for a model or version via Images API.

    Creates a 'user_posts' directory with subfolders per post: user_posts/post_<postId>/
    Each subfolder contains the images and their JSON metadata, plus a post.json summary.

    Args:
        model_id: Model ID
        output_dir: Model output directory
        base_name: Base filename base
        posts_limit: Max number of posts to save (0 = unlimited)
        images_per_post_limit: Max images per post (0 = unlimited)
        session: Optional requests session
        model_version_id: Optional version ID to filter
        user_images_level: NSFW level filter for images endpoint (PG, PG-13, R, X, XXX, ALL)

    Returns:
        Dict[postId, int]: Mapping of postId to number of images saved
    """
    saved_counts: Dict[int, int] = {}
    try:
        session = session or requests.Session()
        posts_root = Path(output_dir) / 'user_posts'
        posts_root.mkdir(parents=True, exist_ok=True)

        # Set browser-like headers similar to fetch_user_images
        session.headers.update({
            'Accept': 'application/json, text/plain, */*',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
            'Referer': 'https://civitai.com/'
        })
        api_token = os.environ.get('CIVITAI_API_TOKEN')
        if api_token:
            session.headers['Authorization'] = f'Bearer {api_token}'

        # Pagination through images endpoint; group by postId
        next_page_url = None
        total_posts_collected = 0
        # NSFW params mapping
        level = (user_images_level or 'ALL').upper()

        # Track groups in memory until we persist respecting limits
        from collections import defaultdict
        groups: DefaultDict[int, List[Dict]] = defaultdict(list)

        # Local cache for model version info to avoid repeated requests
        version_info_cache: Dict[int, Dict[str, str]] = {}

        def get_model_version_info_local(vid: int) -> Optional[Dict[str, str]]:
            try:
                if not vid:
                    return None
                if vid in version_info_cache:
                    return version_info_cache[vid]
                url = f"https://civitai.com/api/v1/model-versions/{int(vid)}"
                resp = session.get(url, timeout=20)
                if resp.status_code != 200:
                    version_info_cache[vid] = {}
                    return None
                data_mv = resp.json() if resp.content else {}
                model_name = (data_mv.get('model') or {}).get('name') if isinstance(data_mv, dict) else None
                version_name = data_mv.get('name') if isinstance(data_mv, dict) else None
                label = f"{model_name} - {version_name}" if model_name and version_name else (version_name or model_name or str(vid))
                result_mv = {
                    'modelName': model_name or '',
                    'versionName': version_name or '',
                    'label': label,
                }
                version_info_cache[vid] = result_mv
                return result_mv
            except Exception:
                return None

        while True:
            if next_page_url:
                url = next_page_url
                params = None
            else:
                params = {
                    'limit': 100,
                }
                if level == 'ALL':
                    params['nsfw'] = 'true'
                elif level == 'PG':
                    params['nsfw'] = 'false'
                else:
                    params['nsfw'] = 'true'
                    params['nsfwLevel'] = level
                if model_version_id:
                    params['modelVersionId'] = model_version_id
                else:
                    params['modelId'] = model_id
                url = 'https://civitai.com/api/v1/images'

            r = session.get(url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"Failed to fetch user posts/images (status {r.status_code})")
                break
            data = r.json() if r.content else {}
            items = data.get('items') if isinstance(data, dict) else []
            metadata = data.get('metadata') if isinstance(data, dict) else {}

            # Fallback to modelId if version returns empty
            if not items and model_version_id and params is None:
                # Already following nextPage cursor; if still empty, stop
                break
            if not items and model_version_id and params is not None and 'modelVersionId' in params:
                # immediate fallback one-time to modelId
                print("No items for modelVersionId; falling back to modelId")
                model_version_id = None
                next_page_url = None
                continue

            if not items:
                break

            # Group by postId
            for it in items:
                pid = it.get('postId')
                if not pid:
                    # Some images might be unattached; skip
                    continue
                if posts_limit and len(groups) >= posts_limit and pid not in groups:
                    # Already reached posts limit; skip new postIds
                    continue
                # Enforce per-post image limit while grouping
                if images_per_post_limit and len(groups[pid]) >= images_per_post_limit:
                    continue
                groups[pid].append(it)

            next_page_url = metadata.get('nextPage') if isinstance(metadata, dict) else None
            # Stop early if we've reached both limits and there is no need to fetch more
            if posts_limit and len(groups) >= posts_limit:
                # Ensure all groups up to limit are filled to images_per_post_limit if more pages exist is optional; we stop here for efficiency
                pass
            if not next_page_url:
                break

        # Persist groups to disk according to limits
        for pid in list(groups.keys())[: (posts_limit or len(groups))]:
            images = groups[pid][: (images_per_post_limit or len(groups[pid]))]
            post_dir = posts_root / f"post_{pid}"
            post_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            post_meta: Dict = {
                'postId': pid,
                'username': images[0].get('username') if images else None,
                'createdAt': images[0].get('createdAt') if images else None,
                'stats': images[0].get('stats') if images else None,
                'imageCount': len(images),
            }
            for idx, item in enumerate(images):
                try:
                    image_url = item.get('url') or item.get('meta', {}).get('url')
                    if not image_url:
                        continue
                    ext = Path(image_url.split('?')[0]).suffix or '.jpeg'
                    filename = f"{sanitize_filename(base_name)}_post_{pid}_{idx}{ext}"
                    target_path = post_dir / filename
                    rr = session.get(image_url, stream=True, timeout=30)
                    if rr.status_code == 200:
                        with open(target_path, 'wb') as f:
                            for chunk in rr.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        # Enrich and save image metadata
                        try:
                            meta = item.get('meta') if isinstance(item, dict) else None
                            if isinstance(meta, dict) and isinstance(meta.get('civitaiResources'), list):
                                enriched = []
                                for r in meta['civitaiResources']:
                                    rr_loc = dict(r) if isinstance(r, dict) else {}
                                    if rr_loc and not rr_loc.get('modelVersionName'):
                                        mvid = rr_loc.get('modelVersionId') or rr_loc.get('id')
                                        if mvid:
                                            info = get_model_version_info_local(int(mvid))
                                            if info:
                                                rr_loc['modelVersionName'] = info.get('label') or rr_loc.get('modelVersionName')
                                                if info.get('modelName'):
                                                    rr_loc['modelName'] = info['modelName']
                                                if info.get('versionName'):
                                                    rr_loc['resolvedVersionName'] = info['versionName']
                                    enriched.append(rr_loc if rr_loc else r)
                                meta['civitaiResources'] = enriched
                        except Exception:
                            pass
                        with open(target_path.with_suffix('.json'), 'w', encoding='utf-8') as jf:
                            json.dump(item, jf, indent=4)
                        saved += 1
                except Exception as ie:
                    print(f"Error downloading post image: {ie}")
                    continue

            # Save post summary
            post_meta['savedImages'] = saved
            try:
                with open(post_dir / 'post.json', 'w', encoding='utf-8') as pf:
                    json.dump(post_meta, pf, indent=4)
            except Exception as je:
                print(f"Error saving post.json for post {pid}: {je}")

            saved_counts[pid] = saved
            total_posts_collected += 1
            if posts_limit and total_posts_collected >= posts_limit:
                break

        print(f"Saved {len(saved_counts)} user posts with per-post images up to limit {images_per_post_limit or 'ALL'}")
        return saved_counts
    except Exception as e:
        print(f"Error fetching user posts: {e}")
        return saved_counts

def download_preview_image(
    image_url: str,
    output_dir: Path,
    base_name: str,
    index: Optional[int] = None,
    is_video: bool = False,
    image_data: Optional[Dict] = None,
    subdir: Optional[str] = None,
):
    """
    Download a preview image or video and save optional metadata.

    Args:
        image_url (str): URL of the image/video to download
        output_dir (Path): Base output directory
        base_name (str): Base name of the safetensors file
        index (int, optional): Image index for multiple images

    Returns:
        Optional[str]: Relative path (to output_dir) of the saved file, or None on failure
    """
    try:
        if not image_url:
            return None

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
    user_images_level: str = 'ALL',
) -> int:
    """
    Fetch user-generated images for a model and save them under user_images subfolder.

    Args:
        model_id: Civitai model ID
        output_dir: Model output directory
        base_name: Base model name (sanitized)
        limit: Number of images to fetch (0 disables)
        session: Optional requests session
        model_version_id: Civitai modelVersionId to filter exactly

    Returns:
        int: Number of images downloaded
    """
    if not limit or limit <= 0:
        return 0

    downloaded = 0
    try:
        session = session or requests.Session()
        # Use browser-like headers to reduce likelihood of being blocked by edge protection
        session.headers.update({
            'Accept': 'application/json, text/plain, */*',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
            'Referer': 'https://civitai.com/'
        })

        # Optional API token via env var
        api_token = os.environ.get('CIVITAI_API_TOKEN')
        if api_token:
            session.headers['Authorization'] = f'Bearer {api_token}'

        # Cap per docs (0..200). We'll page if more are requested.
        total_requested = int(limit)
        per_page = max(1, min(200, total_requested if total_requested > 0 else 100))

        base_url = 'https://civitai.com/api/v1/images'
        print(f"\nFetching user images: {base_url} (limit={total_requested}, per_page={per_page})")

        def fetch_page(url: str, params: Optional[dict] = None, attempt_base_delay: float = 1.5):
            data_local = None
            last_status = None
            for attempt in range(3):
                resp = session.get(url, params=params, timeout=20)
                last_status = resp.status_code
                if resp.status_code != 200:
                    print(f"Error fetching user images (Status code: {resp.status_code}) on attempt {attempt+1}")
                    time.sleep(attempt_base_delay * (attempt + 1))
                    continue
                ctype = resp.headers.get('Content-Type', '')
                if 'application/json' not in ctype:
                    text_head = (resp.text or '')[:160].replace('\n', ' ')
                    print(f"Warning: Expected JSON but got Content-Type '{ctype}' (attempt {attempt+1}). Body: {text_head}")
                    time.sleep(attempt_base_delay * (attempt + 1))
                    continue
                try:
                    data_local = resp.json()
                    break
                except Exception as je:
                    print(f"Warning: Failed to parse user images JSON on attempt {attempt+1}: {je}")
                    time.sleep(attempt_base_delay * (attempt + 1))
                    continue
            return data_local, last_status

        # Prepare first params (prefer modelVersionId)
        params = {
            ('modelVersionId' if model_version_id else 'modelId'): (model_version_id if model_version_id else model_id),
            'limit': per_page,
        }
        # Map browsing level to API params
        level = (user_images_level or 'ALL').upper()
        if level == 'ALL':
            params['nsfw'] = 'true'
        elif level == 'PG':
            params['nsfw'] = 'false'
        else:
            params['nsfw'] = 'true'
            # Pass nsfwLevel for finer control; server may ignore if unknown
            params['nsfwLevel'] = level

        user_subdir = Path(output_dir) / 'user_images'
        user_subdir.mkdir(parents=True, exist_ok=True)

        next_page_url: Optional[str] = None
        tried_fallback = False
        page_index = 0

        while True:
            # Decide request target
            if next_page_url:
                data, last_status = fetch_page(next_page_url, params=None, attempt_base_delay=1.5)
            else:
                data, last_status = fetch_page(base_url, params=params, attempt_base_delay=1.5)

            if data is None:
                print(f"Giving up fetching user images after retries on page {page_index}. Last status: {last_status}")
                break

            # Extract items and metadata
            items = data.get('items') if isinstance(data, dict) else data
            metadata = data.get('metadata', {}) if isinstance(data, dict) else {}

            # Safety filter for modelVersionId (only when not in fallback mode)
            if model_version_id and not tried_fallback and items:
                try:
                    items = [it for it in items if it.get('modelVersionId') == model_version_id]
                except Exception:
                    pass

            if (not items) and (not next_page_url) and model_version_id and not tried_fallback:
                # First page empty: try fallback with modelId
                fb_params = {'modelId': model_id, 'limit': per_page}
                if level == 'ALL':
                    fb_params['nsfw'] = 'true'
                elif level == 'PG':
                    fb_params['nsfw'] = 'false'
                else:
                    fb_params['nsfw'] = 'true'
                    fb_params['nsfwLevel'] = level
                print(f"No items for modelVersionId={model_version_id}. Falling back to modelId with params={fb_params}")
                params = fb_params
                tried_fallback = True
                # Retry this loop iteration with fallback params
                next_page_url = None
                page_index += 1
                continue

            if not items:
                break

            # Download items up to requested total
            for item in items:
                if total_requested and downloaded >= total_requested:
                    break
                try:
                    image_url = item.get('url') or item.get('meta', {}).get('url')
                    if not image_url:
                        continue
                    # Preserve extension including gifs/webp/avif; default to .jpeg
                    ext = Path(image_url.split('?')[0]).suffix
                    if not ext:
                        ext = '.jpeg'
                    filename = f"{sanitize_filename(base_name)}_user_{downloaded}{ext}"
                    target_path = user_subdir / filename

                    r = session.get(image_url, stream=True, timeout=20)
                    if r.status_code == 200:
                        with open(target_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        # Save metadata
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

            if total_requested and downloaded >= total_requested:
                break

            # Determine next page via cursor/nextPage
            next_page_url = metadata.get('nextPage') if isinstance(metadata, dict) else None
            if not next_page_url:
                break
            page_index += 1

        print(f"Downloaded {downloaded} user images")
        return downloaded
    except Exception as e:
        print(f"Error fetching user images: {e}")
        return downloaded

def fetch_model_details(
    model_id: int,
    output_dir: Path,
    safetensors_path: Path,
    session: Optional[requests.Session] = None
) -> bool:
    """
    Fetch model details from Civitai API and save them alongside the model files.

    Args:
        model_id: Civitai model ID
        output_dir: Model output directory
        safetensors_path: Path to the original model file for naming
        session: Optional requests session

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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_data_path = output_path / f"{base_name}_civitai_model.json"

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
    user_images_level: str = 'ALL',
    # New limits for posts
    user_posts_limit: int = 0,
    images_per_post_limit: int = 0,
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
            # Fetch user posts if configured
            if not skip_images and (user_posts_limit or 0):
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
                    fetch_user_posts(
                        model_id=model_id,
                        output_dir=model_output_dir,
                        base_name=sanitize_filename(file_path.stem),
                        posts_limit=(user_posts_limit or 0),
                        images_per_post_limit=images_per_post_limit or 0,
                        session=session,
                        model_version_id=model_version_id,
                        user_images_level=user_images_level,
                    )
                except Exception as e:
                    print(f"Warning: failed to fetch user posts: {e}")
            generate_html_summary(model_output_dir, file_path)
            return True
        else:
            return False
            
    return False