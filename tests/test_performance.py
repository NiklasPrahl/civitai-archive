import pytest
import time
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from civitai_manager.src.core.metadata_manager import process_directory, find_safetensors_files
from civitai_manager.src.utils.process_manager import ProcessManager
from civitai_manager.src.utils.string_utils import calculate_sha256

def create_dummy_safetensors(path: Path, size_mb: int = 1):
    """Create a dummy safetensors file of specified size"""
    import json
    # Create header with test metadata
    metadata = json.dumps({"test": "data"}).encode('utf-8')
    padding_length = 84 - len(metadata)
    header = metadata + (b' ' * padding_length)
    header_length = len(header)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        # Write header length (8 bytes, little-endian)
        f.write(header_length.to_bytes(8, 'little'))
        # Write header
        f.write(header)
        # Write dummy tensor data to reach desired size
        remaining_size = (size_mb * 1024 * 1024) - header_length - 8
        chunk_size = 1024 * 1024  # 1MB chunks
        while remaining_size > 0:
            write_size = min(chunk_size, remaining_size)
            f.write(b'0' * write_size)  # Use '0' instead of 'x' for consistent hash
            remaining_size -= write_size

def test_large_file_processing(requests_mock, temp_dir, mock_civitai_responses):
    """Test processing of large files"""
    models_dir = temp_dir / 'models'
    output_dir = temp_dir / 'output'
    models_dir.mkdir()
    output_dir.mkdir()
    
    # Create a 100MB test file
    test_file = models_dir / 'large_model.safetensors'
    create_dummy_safetensors(test_file, size_mb=100)
    hash_value = calculate_sha256(test_file)
    model_id = mock_civitai_responses['version']['modelId']

    requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}", json=mock_civitai_responses['version'])
    requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}", json=mock_civitai_responses['model'])

    start_time = time.time()
    processed, failed, skipped = process_directory(
        models_dir,
        output_dir,
        no_timeout=True,
        skip_images=True
    )
    processing_time = time.time() - start_time
    
    assert processed == 1
    assert failed == 0
    assert skipped == 0
    assert processing_time < 30  # Should process within 30 seconds
    
    # Check memory usage didn't spike
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    assert memory_mb < 200  # Memory usage should stay under 200MB

def test_concurrent_processing(requests_mock, temp_dir, mock_civitai_responses):
    """Test processing multiple files concurrently"""
    models_dir = temp_dir / 'models'
    output_dir = temp_dir / 'output'
    models_dir.mkdir()
    output_dir.mkdir()

    # Create multiple test files
    test_files = []
    for i in range(5):
        file_path = models_dir / f'model_{i}.safetensors'
        create_dummy_safetensors(file_path, size_mb=10)
        test_files.append(file_path)
        hash_value = calculate_sha256(file_path)
        model_id = mock_civitai_responses['version']['modelId']
        requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}", json=mock_civitai_responses['version'])
        requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}", json=mock_civitai_responses['model'])

    # Process files concurrently
    processed, failed, skipped = process_directory(
        models_dir,
        output_dir,
        no_timeout=True,
        skip_images=True,
        max_workers=3
    )
    assert processed == 5
    assert failed == 0
    assert skipped == 0

    # Check all files were processed
    processed_files = list(output_dir.glob('**/*_metadata.json'))
    assert len(processed_files) == len(test_files)

def test_stress_test(requests_mock, temp_dir, mock_civitai_responses):
    """Stress test with many small files"""
    models_dir = temp_dir / 'models'
    output_dir = temp_dir / 'output'
    models_dir.mkdir()
    output_dir.mkdir()
    
    # Create many small test files
    num_files = 50
    for i in range(num_files):
        file_path = models_dir / f'model_{i}.safetensors'
        create_dummy_safetensors(file_path, size_mb=1)
        hash_value = calculate_sha256(file_path)
        model_id = mock_civitai_responses['version']['modelId']
        requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}", json=mock_civitai_responses['version'])
        requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}", json=mock_civitai_responses['model'])

    start_time = time.time()
    processed, failed, skipped = process_directory(
        models_dir,
        output_dir,
        no_timeout=True,
        skip_images=True,
        max_workers=4
    )
    processing_time = time.time() - start_time
    
    assert processed == num_files
    assert failed == 0
    assert skipped == 0
    # Should process at least 2 files per second on average
    assert processing_time < (num_files / 2)
    
    # Check all files were processed
    processed_files = list(output_dir.glob('**/*_metadata.json'))
    assert len(processed_files) == num_files

def test_memory_leak_check(requests_mock, temp_dir, mock_civitai_responses):
    """Test for memory leaks during repeated processing"""
    models_dir = temp_dir / 'models'
    output_dir = temp_dir / 'output'
    models_dir.mkdir()
    output_dir.mkdir()
    
    # Create test file
    test_file = models_dir / 'test_model.safetensors'
    create_dummy_safetensors(test_file, size_mb=50)
    hash_value = calculate_sha256(test_file)
    model_id = mock_civitai_responses['version']['modelId']
    requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}", json=mock_civitai_responses['version'])
    requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}", json=mock_civitai_responses['model'])

    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Process the same file multiple times
    for _ in range(10):
        processed, failed, skipped = process_directory(
            models_dir,
            output_dir,
            no_timeout=True,
            skip_images=True
        )
        assert processed == 1
        assert failed == 0
        assert skipped == 0
        
        # Clear output directory
        for file in output_dir.glob('**/*'):
            if file.is_file():
                file.unlink()
        for dir in output_dir.glob('**/*'):
            if dir.is_dir():
                dir.rmdir()

    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
    
    # Memory increase should be minimal
    assert memory_increase < 50  # Less than 50MB increase