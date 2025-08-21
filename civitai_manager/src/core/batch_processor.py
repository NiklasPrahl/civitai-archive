import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import requests

from .file_processor import process_single_file

@dataclass
class ProcessingMetrics:
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    start_time: float = 0.0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def files_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0
        return self.processed_files / self.elapsed_time

class BatchProcessor:
    def __init__(self, max_workers: int = 4, download_all_images: bool = False, skip_images: bool = False, html_only: bool = False, only_update: bool = False, user_images_limit: int = 0, user_images_level: str = 'ALL'):
        self.max_workers = max_workers
        self.metrics = ProcessingMetrics()
        self.session = requests.Session()  # Reuse HTTP session
        self._cancel = threading.Event()
        self.download_all_images = download_all_images
        self.skip_images = skip_images
        self.html_only = html_only
        self.only_update = only_update
        self.user_images_limit = user_images_limit
        self.user_images_level = user_images_level
        
    def process_files(self, files: List[Path], output_dir: Path) -> ProcessingMetrics:
        """Process multiple files concurrently"""
        self.metrics = ProcessingMetrics()
        self.metrics.total_files = len(files)
        self.metrics.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in files:
                if self._cancel.is_set():
                    break
                futures.append(
                    executor.submit(
                        process_single_file,
                        file,
                        output_dir,
                        self.download_all_images,
                        self.skip_images,
                        self.html_only,
                        self.only_update,
                        self.session,
                        self.user_images_limit,
                        self.user_images_level,
                    )
                )
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        self.metrics.processed_files += 1
                    else:
                        self.metrics.failed_files += 1
                except Exception as e:
                    logging.error(f"Error processing file: {e}")
                    self.metrics.failed_files += 1
                    
        return self.metrics
            
    def cancel(self):
        """Cancel ongoing processing"""
        self._cancel.set()
        
    def reset(self):
        """Reset for new processing run"""
        self._cancel.clear()
        self.metrics = ProcessingMetrics()