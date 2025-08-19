from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import threading
import logging
from queue import Queue

@dataclass
class ProcessStatus:
    filename: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0

class ProcessManager:
    def __init__(self):
        self._processes: Dict[str, ProcessStatus] = {}
        self._lock = threading.Lock()
        self._queue = Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        self._max_history = 100
        
    def add_process(self, filename: str) -> str:
        """Add a new process to track"""
        with self._lock:
            process_id = filename
            self._processes[process_id] = ProcessStatus(
                filename=filename,
                status='pending',
                start_time=datetime.now()
            )
            # Cleanup old completed processes if needed
            if len(self._processes) > self._max_history:
                self._cleanup_old_processes()
            return process_id
            
    def update_status(self, process_id: str, status: str, error: Optional[str] = None, progress: float = None):
        """Update process status"""
        with self._lock:
            if process_id in self._processes:
                process = self._processes[process_id]
                process.status = status
                if error:
                    process.error = error
                if progress is not None:
                    process.progress = progress
                if status in ['completed', 'failed']:
                    process.end_time = datetime.now()
                    
    def get_status(self, process_id: str) -> Optional[ProcessStatus]:
        """Get current status of a process"""
        with self._lock:
            return self._processes.get(process_id)
            
    def get_all_active(self) -> List[ProcessStatus]:
        """Get all active processes"""
        with self._lock:
            return [p for p in self._processes.values() 
                   if p.status in ['pending', 'processing']]
                   
    def get_recent_completed(self, limit: int = 10) -> List[ProcessStatus]:
        """Get recently completed processes"""
        with self._lock:
            completed = [p for p in self._processes.values() 
                       if p.status in ['completed', 'failed']]
            completed.sort(key=lambda x: x.end_time or datetime.min, reverse=True)
            return completed[:limit]
            
    def _cleanup_old_processes(self):
        """Remove old completed processes"""
        processes = list(self._processes.items())
        processes.sort(key=lambda x: x[1].end_time or datetime.max, reverse=True)
        
        # Keep only the most recent ones
        self._processes = dict(processes[:self._max_history])
        
    def queue_process(self, func, *args, **kwargs):
        """Queue a process for execution"""
        self._queue.put((func, args, kwargs))
        
    def _process_queue(self):
        """Worker thread to process the queue"""
        while True:
            try:
                func, args, kwargs = self._queue.get()
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in queued process: {e}")
            except Exception as e:
                logging.error(f"Error in process queue: {e}")
            finally:
                self._queue.task_done()
