from pathlib import Path
import json
from datetime import datetime

class ProcessedFilesManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.processed_file = self.output_dir / 'processed_files.json'
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self):
        """Load the list of processed files from JSON"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return {'files': [], 'last_update': None}
        return {'files': [], 'last_update': None}

    def save_processed_files(self):
        """Save the current list of processed files"""
        with open(self.processed_file, 'w') as f:
            self.processed_files['last_update'] = datetime.now().isoformat()
            json.dump(self.processed_files, f, indent=4)

    def is_file_processed(self, file_path):
        """Check if a file has been processed before"""
        return str(file_path) in self.processed_files['files']

    def add_processed_file(self, file_path):
        """Add a file to the processed list"""
        if str(file_path) not in self.processed_files['files']:
            self.processed_files['files'].append(str(file_path))

    def remove_processed_file(self, file_path):
        """Remove a file from the processed list"""
        file_path_str = str(file_path)
        if file_path_str in self.processed_files['files']:
            self.processed_files['files'].remove(file_path_str)

    def _find_safetensors_files(self, directory_path):
        """Find .safetensors files recursively, following symbolic links"""
        import os
        safetensors_files = []
        for root, dirs, files in os.walk(directory_path, followlinks=True):
            for file in files:
                if file.endswith('.safetensors'):
                    safetensors_files.append(Path(root) / file)
        return safetensors_files

    def get_new_files(self, directory_path):
        """Get list of new safetensors files that haven't been processed"""
        all_files = self._find_safetensors_files(directory_path)
        return [f for f in all_files if not self.is_file_processed(f)]
    
    def update_timestamp(self):
        """Update the last_update timestamp without modifying the files list"""
        self.processed_files['last_update'] = datetime.now().isoformat()
        self.save_processed_files()
