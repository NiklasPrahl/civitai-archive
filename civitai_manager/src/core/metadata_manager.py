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
from concurrent.futures import ThreadPoolExecutor, as_completed

from .batch_processor import BatchProcessor
from .file_processor import process_single_file


from ..utils.file_tracker import ProcessedFilesManager
from ..utils.string_utils import sanitize_filename, calculate_sha256
from ..utils.html_generators.model_page import generate_html_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Please install it using:")
    print("pip install requests")
    sys.exit(1)

def find_safetensors_files(directory_path):
    """Find model files (safetensors, ckpt, pt, pth, bin) recursively.

    Kept function name for backward compatibility.
    """
    model_files = []
    exts = ('.safetensors', '.ckpt', '.pt', '.pth', '.bin')
    for root, dirs, files in os.walk(directory_path, followlinks=True):
        for file in files:
            if file.lower().endswith(exts):
                model_files.append(Path(root) / file)
    return model_files

def get_output_path(clean=False):
    """
    Get output path from user and create necessary directories.
    If no path is provided (empty input), use current directory.
    
    Returns:
        Path: Base output directory path
    """
    while True:
        if clean:
            output_path = input("Enter the path you want to clean (press Enter for current directory): ").strip()
        else:
            output_path = input("Enter the path where you want to save the exported files (press Enter for current directory): ").strip()
        
        if not output_path:
            path = Path.cwd() / 'output'
            print(f"Using current directory: {path}")
        else:
            path = Path(output_path)
        
        if not path.exists():
            try:
                create = input(f"Directory {path} doesn't exist. Create it? (y/n): ").lower()
                if create == 'y':
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    continue
            except Exception as e:
                print(f"Error creating directory: {str(e)}")
                continue
        
        if not os.access(path, os.W_OK):
            print("Error: No write permission for this directory")
            continue
            
        return path

def generate_image_json_files(base_output_path):
    """
    Generate JSON files for all preview images from existing model version data
    
    Args:
        base_output_path (Path): Base output directory path
    """
    print("\nGenerating JSON files for preview images...")
    
    version_files = list(Path(base_output_path).glob('*/*_civitai_model_version.json'))
    total_generated = 0
    
    for version_file in version_files:
        try:
            with open(version_file, 'r', encoding='utf-8') as f:
                version_data = json.load(f)
            
            model_dir = version_file.parent
            
            if 'images' in version_data:
                for i, image_data in enumerate(version_data['images']):
                    ext = '.mp4' if image_data.get('type') == 'video' else '.jpeg'
                    preview_file = model_dir / f"{model_dir.name}_preview_{i}{ext}"
                    
                    if preview_file.exists():
                        json_file = preview_file.with_suffix('.json')
                        
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(image_data, f, indent=4)
                            total_generated += 1
                            
        except Exception as e:
            print(f"Error processing {version_file}: {str(e)}")
            continue
    
    print(f"\nGenerated {total_generated} JSON files for preview images")
    return True

def find_duplicate_models(directory_path, base_output_path):
    """
    Find models with duplicate hashes
    
    Args:
        directory_path (Path): Directory containing safetensors files
        base_output_path (Path): Base output directory path
        
    Returns:
        dict: Dictionary mapping hashes to lists of model info
    """
    hash_map = {}
    
    # Scan all processed models
    for model_dir in base_output_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        hash_file = model_dir / f"{model_dir.name}_hash.json"
        if not hash_file.exists():
            continue
            
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                hash_data = json.load(f)
                hash_value = hash_data.get('hash_value')
                if not hash_value:
                    continue
                    
                # Find corresponding safetensors file
                safetensors_file = None
                for file in find_safetensors_files(directory_path):
                    if file.stem == model_dir.name:
                        safetensors_file = file
                        break
                
                if not safetensors_file:
                    continue
                    
                if hash_value not in hash_map:
                    hash_map[hash_value] = []
                    
                hash_map[hash_value].append({
                    'model_dir': model_dir,
                    'safetensors_file': safetensors_file,
                    'processed_time': hash_data.get('timestamp')
                })
                
        except Exception as e:
            print(f"Error reading hash file {hash_file}: {e}")
            continue
            
    return {k: v for k, v in hash_map.items() if len(v) > 1}

def clean_output_directory(directory_path, base_output_path):
    """
    Clean up output directory by removing data for models that no longer exist
    
    Args:
        directory_path (Path): Directory containing the safetensors files
        base_output_path (Path): Base output directory path
    """

    print("\nStarting cleanup process (duplicates)...")

    # Handle duplicates
    duplicates = find_duplicate_models(directory_path, base_output_path)
    duplicate_file = None
    
    if duplicates:
        duplicate_file = base_output_path / "duplicate_models.txt"
        with open(duplicate_file, 'w', encoding='utf-8') as f:
            f.write("# Duplicate models found in input directory\n")
            f.write("# Format: Hash | Kept Model | Removed Duplicates\n")
            f.write("# This file is automatically updated when running --clean\n\n")
            
            for hash_value, models in duplicates.items():
                sorted_models = sorted(models, 
                    key=lambda x: x['processed_time'] if x['processed_time'] else '',
                    reverse=True
                )
                
                kept_model = sorted_models[0]
                removed_models = sorted_models[1:]
                
                f.write(f"Hash: {hash_value}\n")
                f.write(f'Kept: {kept_model["safetensors_file"]}\n')
                f.write("Removed:\n")
                
                for model in removed_models:
                    f.write(f'  - {model["safetensors_file"]}\n')
                    print(f'Removing duplicate model: {model["model_dir"].name}')
                    try:
                        shutil.rmtree(model['model_dir'])
                    except Exception as e:
                        print(f"Error removing directory {model['model_dir']}: {e}")
                f.write("\n")    
 
        print(f"\nDuplicate models list saved to {duplicate_file}")
    else:
        print("\nNo duplicates to remove")

    print("\nStarting cleanup process (removed models)...")
    existing_models = {
        Path(file).stem
        for file in find_safetensors_files(directory_path)
    }
    
    # Check each directory in output
    output_dirs = [d for d in base_output_path.iterdir() if d.is_dir()]
    cleaned_dirs = []
    
    for output_dir in output_dirs:
        if output_dir.name not in existing_models:
            print(f"Removing directory: {output_dir.name} (model not found)")
            try:
                shutil.rmtree(output_dir)
                cleaned_dirs.append(str(output_dir))
            except Exception as e:
                print(f"Error removing directory {output_dir}: {e}")
    
    # Update processed_files.json
    if cleaned_dirs:
        files_manager = ProcessedFilesManager(base_output_path)
        new_processed_files = [
            f for f in files_manager.processed_files['files']
            if Path(f).stem in existing_models
        ]
        files_manager.processed_files['files'] = new_processed_files
        files_manager.save_processed_files()
        
        print(f"\nCleaned up {len(cleaned_dirs)} directories")
    else:
        print("\nNo directories to clean")
    
    return True

def process_directory(
    directory_path: Path,
    base_output_path: Path,
    no_timeout: bool = False,
    download_all_images: bool = False,
    skip_images: bool = False,
    only_new: bool = False,
    html_only: bool = False,
    only_update: bool = False,
    skip_missing: bool = False,
    max_workers: int = 4,
    user_images_limit: int = 0,
    cancel_flag = None
) -> Tuple[int, int, int]:
    """
    Process all safetensors files in a directory using parallel processing
            
    Args:
        directory_path: Path to the directory containing safetensors files
        base_output_path: Base path for output
        no_timeout: If True, disable timeout between files
        download_all_images: Whether to download all available preview images
        skip_images: Whether to skip downloading images completely
        only_new: Whether to only process new models
        html_only: Whether to only generate HTML files
        only_update: Whether to only update existing processed files
        skip_missing: Whether to skip missing files
        max_workers: Maximum number of parallel workers
        cancel_flag: Optional flag to cancel processing
        
    Returns:
        Tuple containing (processed_count, failed_count, skipped_count)
    """
    if not directory_path.exists():
        logging.error(f"Directory {directory_path} not found")
        return (0, 0, 0)
        
    # Initialize the batch processor
    processor = BatchProcessor(
        max_workers=max_workers,
        download_all_images=download_all_images,
        skip_images=skip_images,
        html_only=html_only,
        only_update=only_update,
        user_images_limit=user_images_limit,
    )
    metrics = None

    # Get the list of files to process
    files_manager = None if html_only else ProcessedFilesManager(base_output_path)
    
    try:
        if only_new and not html_only:
            safetensors_files = files_manager.get_new_files(directory_path)
            if not safetensors_files:
                logging.info("No new files to process")
                return (0, 0, 0)
            
            if skip_missing:
                # Read missing models file
                missing_file = Path(base_output_path) / 'missing_from_civitai.txt'
                missing_models = set()
                if missing_file.exists():
                    with open(missing_file, 'r', encoding='utf-8') as f:
                        missing_models = {
                            line.strip().split(' | ')[-1]
                            for line in f
                            if line.strip() and not line.startswith('#')
                        }
                        
                # Filter out previously missing models
                safetensors_files = [
                    f for f in safetensors_files 
                    if f.name not in missing_models
                ]
                if not safetensors_files:
                    logging.info("No new non-missing files to process")
                    return (0, 0, 0)

            logging.info(f"Found {len(safetensors_files)} new model files")
            
        elif only_update:
            # Only get previously processed files
            all_files = find_safetensors_files(directory_path)
            safetensors_files = [
                file_path for file_path in all_files
                if (Path(base_output_path) / file_path.stem / f"{file_path.stem}_hash.json").exists()
            ]
            logging.info(f"Found {len(safetensors_files)} previously processed files")
            
        else:
            safetensors_files = find_safetensors_files(directory_path)
            if not safetensors_files:
                logging.warning(f"No model files found in {directory_path}")
                return (0, 0, 0)
            logging.info(f"Found {len(safetensors_files)} model files")
        
        if html_only:
            logging.info("HTML only mode: Skipping data fetching")
            
        # Process files in batches
        metrics = processor.process_files(safetensors_files, base_output_path)
        
        # Check if processing was cancelled
        if cancel_flag and cancel_flag.is_set():
            logging.info("Processing cancelled by user")
            processor.cancel()
            processor.cancel()
        
        # Update processed files tracking
        if not (html_only or only_update):
            for file_path in safetensors_files[:metrics.processed_files]:
                files_manager.add_processed_file(file_path)
            files_manager.save_processed_files()
            
        # Log final statistics
        elapsed = metrics.elapsed_time
        logging.info(f"\nProcessing completed in {elapsed:.2f} seconds")
        logging.info(f"Files processed: {metrics.processed_files}/{metrics.total_files}")
        logging.info(f"Failed: {metrics.failed_files}")
        logging.info(f"Skipped: {metrics.skipped_files}")
        logging.info(f"Average speed: {metrics.files_per_second:.2f} files/second")
        
        return (metrics.processed_files, metrics.failed_files, metrics.skipped_files)
        
    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}")
        if metrics:
            return (metrics.processed_files, metrics.failed_files, metrics.skipped_files)
        return (0, 0, 0)