from pathlib import Path
import json
import os
from datetime import datetime
import shutil

class ProcessedFilesManager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.processed_file = self.output_dir / 'processed_files.json'
        self.processed_files = self._load_processed_files()
        self.cleanup_threshold = 1000  # Maximum number of entries before cleanup

    def _load_processed_files(self):
        """Load the list of processed files from JSON"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r') as f:
                    data = json.load(f)
                    # Convert old format to new format if necessary
                    if isinstance(data, dict) and 'files' in data:
                        files = []
                        for f in data['files']:
                            if isinstance(f, dict):
                                # Already in new format
                                path = f['path']
                            else:
                                # Old format - just a path string
                                path = f
                            files.append({
                                'path': path,
                                'last_seen': datetime.now().isoformat(),
                                'still_exists': os.path.exists(path)
                            })
                        return {
                            'files': files,
                            'last_update': data.get('last_update', datetime.now().isoformat())
                        }
                    return data
            except (FileNotFoundError, json.JSONDecodeError):
                return {'files': [], 'last_update': None}
        return {'files': [], 'last_update': None}
        
    def cleanup_old_entries(self):
        """Remove entries for files that no longer exist"""
        current_files = []
        for entry in self.processed_files['files']:
            if os.path.exists(entry['path']):
                entry['still_exists'] = True
                entry['last_seen'] = datetime.now().isoformat()
                current_files.append(entry)
            elif not entry['still_exists']:  # Remove if marked as non-existent in previous run
                continue
            else:  # Mark as potentially removable in next run
                entry['still_exists'] = False
                current_files.append(entry)
                
        self.processed_files['files'] = current_files

    def save_processed_files(self):
        """Save the current list of processed files"""
        # Check if we need to clean up
        if len(self.processed_files['files']) > self.cleanup_threshold:
            self.cleanup_old_entries()
            
        # Create backup before saving
        if self.processed_file.exists():
            backup_path = self.output_dir / f'processed_files.{datetime.now().strftime("%Y%m%d")}.bak'
            shutil.copy2(self.processed_file, backup_path)
            
            # Keep only last 5 backups
            backups = sorted(self.output_dir.glob('processed_files.*.bak'))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup.unlink()
        
        with open(self.processed_file, 'w') as f:
            self.processed_files['last_update'] = datetime.now().isoformat()
            json.dump(self.processed_files, f, indent=4)

    def is_file_processed(self, file_path):
        """Check if a file has been processed before"""
        str_path = str(file_path)
        return any(entry['path'] == str_path and entry['still_exists'] 
                   for entry in self.processed_files['files'])

    def add_processed_file(self, file_path):
        """Add a file to the processed list with metadata"""
        file_path_str = str(file_path)
        # Check if entry already exists
        for entry in self.processed_files['files']:
            if entry['path'] == file_path_str:
                entry['last_seen'] = datetime.now().isoformat()
                entry['still_exists'] = True
                return
                
        # Add new entry
        self.processed_files['files'].append({
            'path': file_path_str,
            'last_seen': datetime.now().isoformat(),
            'still_exists': True,
            'first_processed': datetime.now().isoformat()
        })

    def remove_processed_file(self, file_path):
        """Remove a file from the processed list"""
        file_path_str = str(file_path)
        self.processed_files['files'] = [
            entry for entry in self.processed_files['files'] 
            if entry['path'] != file_path_str
        ]

    def get_processing_stats(self):
        """Get statistics about processed files"""
        now = datetime.now()
        stats = {
            'total_files': len(self.processed_files['files']),
            'existing_files': sum(1 for f in self.processed_files['files'] if f['still_exists']),
            'missing_files': sum(1 for f in self.processed_files['files'] if not f['still_exists']),
            'last_update': self.processed_files['last_update'],
        }
        return stats

    def _find_safetensors_files(self, directory_path):
        """Find .safetensors files recursively, following symbolic links"""
        safetensors_files = []
        try:
            for root, dirs, files in os.walk(directory_path, followlinks=True):
                for file in files:
                    if file.lower().endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.bin')):
                        file_path = Path(root) / file
                        if file_path.exists():  # Double-check in case of broken symlinks
                            safetensors_files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory {directory_path}: {e}")
        return safetensors_files

    def get_new_files(self, directory_path):
        """Get list of new safetensors files that haven't been processed"""
        all_files = self._find_safetensors_files(directory_path)
        return [f for f in all_files if not self.is_file_processed(f)]
    
    def update_timestamp(self):
        """Update the last_update timestamp without modifying the files list"""
        self.processed_files['last_update'] = datetime.now().isoformat()
        self.save_processed_files()
