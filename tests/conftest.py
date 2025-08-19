import pytest
from pathlib import Path
import os
import tempfile
import shutil
import json

# Setup test paths
@pytest.fixture(scope="session")
def app():
    """Create a single Flask app instance for the test session"""
    from civitai_manager.web_app import create_app
    
    app = create_app()
    app.config.update(
        TESTING=True,
        SECRET_KEY='test-key',
        WTF_CSRF_ENABLED=True,
        WTF_CSRF_CHECK_DEFAULT=True,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024
    )
    return app

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def test_config(app, temp_dir):
    """Create a test configuration and configure the app"""
    config = {
        'models_directory': str(temp_dir / 'models'),
        'output_directory': str(temp_dir / 'output'),
        'download_all_images': False,
        'skip_images': True,
        'notimeout': True
    }
    
    # Create required directories
    Path(config['models_directory']).mkdir(parents=True, exist_ok=True)
    Path(config['output_directory']).mkdir(parents=True, exist_ok=True)
    (temp_dir / 'uploads').mkdir(parents=True, exist_ok=True)

    # Update app config
    app.config.update(
        CONFIG_FILE=str(temp_dir / 'config.json'),
        MODELS_DIR=config['models_directory'],
        OUTPUT_DIR=config['output_directory'],
        UPLOAD_FOLDER=str(temp_dir / 'uploads')
    )
    
    # Write config file
    with open(app.config['CONFIG_FILE'], 'w') as f:
        json.dump(config, f, indent=4)
        
    return config

@pytest.fixture
def sample_safetensors(temp_dir):
    """Create a sample safetensors file for testing"""
    model_dir = temp_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    test_file = model_dir / 'test_model.safetensors'
    with open(test_file, 'wb') as f:
        # Write dummy header (8 bytes for length + some metadata)
        header_length = 100
        f.write(header_length.to_bytes(8, 'little'))
        f.write(b'{"metadata": {"test": "data"}}' + b' ' * 70)  # Pad to header_length
        # Write some dummy tensor data
        f.write(b'x' * 1000)
    
    return test_file

@pytest.fixture
def mock_civitai_responses():
    """Mock responses for Civitai API calls"""
    return {
        'model': {
            'name': 'Test Model',
            'type': 'Checkpoint',
            'description': 'Test model description',
            'id': 12345,
            'modelVersions': [
                {
                    'id': 12345,
                    'name': 'Test Model v1.0',
                    'baseModel': 'SD 1.5',
                    'images': [
                        {'type': 'image', 'url': 'https://example.com/preview1.jpg'},
                        {'type': 'image', 'url': 'https://example.com/preview2.jpg'}
                    ],
                    'files': [
                        {
                            'sizeKB': 1024
                        }
                    ]
                }
            ]
        },
        'version': {
            'name': 'Test Model v1.0',
            'modelId': 12345,
            'id': 12345,
            'baseModel': 'SD 1.5',
            'images': [
                {'type': 'image', 'url': 'https://example.com/preview1.jpg'},
                {'type': 'image', 'url': 'https://example.com/preview2.jpg'}
            ],
            'files': [
                {
                    'sizeKB': 1024
                }
            ]
        }
    }

@pytest.fixture
def client(app, test_config):
    """Create a test client for the Flask app"""
    return app.test_client()