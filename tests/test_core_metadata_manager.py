import pytest
from pathlib import Path
import hashlib
from civitai_manager.src.core.file_processor import (
    extract_metadata,
    extract_hash,
    fetch_version_data,
    fetch_model_details,
    process_single_file
)

def test_extract_metadata(temp_dir, sample_safetensors):
    """Test metadata extraction from safetensors file"""
    output_dir = temp_dir / 'output'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test successful metadata extraction
    result = extract_metadata(sample_safetensors, output_dir)
    assert result is True
    
    # Check if metadata file was created
    metadata_file = output_dir / f"{sample_safetensors.stem}_metadata.json"
    assert metadata_file.exists()
    
    # Test with non-existent file
    non_existent = temp_dir / 'nonexistent.safetensors'
    result = extract_metadata(non_existent, output_dir)
    assert result is False
    
    # Test with invalid file extension
    invalid_file = temp_dir / 'test.txt'
    invalid_file.touch()
    result = extract_metadata(invalid_file, output_dir)
    assert result is False

def test_extract_hash(temp_dir, sample_safetensors):
    """Test hash extraction from safetensors file"""
    output_dir = temp_dir / 'output'
    
    # Test successful hash extraction
    hash_value = extract_hash(sample_safetensors, output_dir)
    assert hash_value is not None
    assert len(hash_value) == 64  # SHA-256 hash length
    
    # Verify the hash is correct
    with open(sample_safetensors, 'rb') as f:
        content = f.read()
        expected_hash = hashlib.sha256(content).hexdigest()
    assert hash_value == expected_hash
    
    # Check if hash file was created
    hash_file = output_dir / f"{sample_safetensors.stem}_hash.json"
    assert hash_file.exists()
    
    # Test with non-existent file
    non_existent = temp_dir / 'nonexistent.safetensors'
    hash_value = extract_hash(non_existent, output_dir)
    assert hash_value is None

def test_fetch_version_data(requests_mock, temp_dir, sample_safetensors, mock_civitai_responses):
    """Test fetching version data from Civitai API"""
    output_dir = temp_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    base_path = temp_dir
    
    # Mock successful API call
    requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/dummy_hash", json=mock_civitai_responses['version'])
    model_id = fetch_version_data(
        "dummy_hash",
        output_dir,
        base_path,
        sample_safetensors,
        skip_images=True
    )
    assert model_id == mock_civitai_responses['version']['modelId']
    version_file = output_dir / f"{sample_safetensors.stem}_civitai_model_version.json"
    assert version_file.exists()
    
    # Mock API error
    requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/dummy_hash_error", status_code=404)
    model_id = fetch_version_data(
        "dummy_hash_error",
        output_dir,
        base_path,
        sample_safetensors,
        skip_images=True
    )
    assert model_id is None

def test_fetch_model_details(requests_mock, temp_dir, sample_safetensors, mock_civitai_responses):
    """Test fetching model details from Civitai API"""
    output_dir = temp_dir / 'output'
    model_id = mock_civitai_responses['version']['modelId']

    # Mock successful API call
    requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}", json=mock_civitai_responses['model'])
    success = fetch_model_details(
        model_id,
        output_dir,
        sample_safetensors
    )
    assert success is True
    model_file = output_dir / f"{sample_safetensors.stem}_civitai_model.json"
    assert model_file.exists()
    
    # Mock API error
    requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}_error", status_code=404)
    success = fetch_model_details(
        f"{model_id}_error",
        output_dir,
        sample_safetensors
    )
    assert success is False

def test_process_single_file(requests_mock, temp_dir, sample_safetensors, mock_civitai_responses):
    """Test complete file processing workflow"""
    output_dir = temp_dir / 'output'
    hash_value = 'a066e21db2e5067a315cb393f1b80df97f3977b22c6cd09366acda5979c3b037'
    model_id = mock_civitai_responses['version']['modelId']

    # Mock API calls
    requests_mock.get(f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}", json=mock_civitai_responses['version'])
    requests_mock.get(f"https://civitai.com/api/v1/models/{model_id}", json=mock_civitai_responses['model'])
    
    # Test successful processing
    success = process_single_file(
        sample_safetensors,
        output_dir,
        skip_images=True
    )
    
    assert success is True
    
    # Check all expected files were created
    base_name = sample_safetensors.stem
    model_output_dir = output_dir / base_name
    assert (model_output_dir / f"{base_name}_metadata.json").exists()
    assert (model_output_dir / f"{base_name}_hash.json").exists()
    assert (model_output_dir / f"{base_name}_civitai_model_version.json").exists()
    assert (model_output_dir / f"{base_name}_civitai_model.json").exists()
    
    # Test with non-existent file
    non_existent = temp_dir / 'nonexistent.safetensors'
    success = process_single_file(
        non_existent,
        output_dir,
        skip_images=True
    )
    assert success is False
