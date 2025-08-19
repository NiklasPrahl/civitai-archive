import pytest
from flask import url_for
import json
from pathlib import Path
from civitai_manager.web_app import app
import os

@pytest.fixture
def client(test_config, tmp_path):
    """Create a test client for the Flask app"""
    config_dir = tmp_path / 'config'
    models_dir = tmp_path / 'models'
    output_dir = tmp_path / 'output'
    config_file = config_dir / 'config.json'
    
    # Create necessary directories
    config_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Write configuration
    config_data = {
        'models_directory': str(models_dir),
        'output_directory': str(output_dir),
        'download_all_images': False,
        'skip_images': True,
        'notimeout': True
    }
    config_file.write_text(json.dumps(config_data))
    
    app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,  # Disable CSRF for testing
        CONFIG_FILE=str(config_file),
        MODELS_DIR=str(models_dir),
        OUTPUT_DIR=str(output_dir)
    )
    
    # Configure app with test settings
    with app.test_client() as test_client:
        with app.app_context():
            # Initialize app with the test configuration
            app.config.update(config_data)
            yield test_client

def test_index_redirect_to_settings(client):
    """Test index route redirects to settings when not configured"""
    from civitai_manager.web_app import app
    app.config['CONFIG_FILE'] = '/nonexistent/config.json'
    app.config['MODELS_DIR'] = ''
    app.config['OUTPUT_DIR'] = ''
    
    response = client.get('/')
    assert response.status_code == 302
    assert '/settings' in response.location

def test_index_with_config(client, test_config, mocker):
    """Test index route with configuration"""
    # Mock get_models_info to return test data
    mock_models = [
        {
            'name': 'Test Model',
            'type': 'Checkpoint',
            'preview_image_url': '/static/test.jpg',
            'files': ['test.safetensors']
        }
    ]
    mocker.patch('civitai_manager.web_app.get_models_info', return_value=mock_models)
    
    response = client.get('/')
    assert response.status_code == 200
    assert b'Test Model' in response.data

def test_settings_page(client):
    """Test settings page"""
    response = client.get('/settings')
    assert response.status_code == 200
    assert b'Models Directory' in response.data
    assert b'Output Directory' in response.data

def test_save_settings(client, test_config):
    """Test saving settings"""
    response = client.post('/settings', data={
        'models_directory': test_config['models_directory'],
        'output_directory': test_config['output_directory'],
        'download_all_images': test_config['download_all_images'],
        'skip_images': test_config['skip_images'],
        'notimeout': test_config['notimeout']
    })
    assert response.status_code == 302  # Redirect after save
    
    # Verify config was saved
    with open(app.config['CONFIG_FILE'], 'r') as f:
        saved_config = json.load(f)
        assert Path(saved_config['models_directory']).resolve() == Path(test_config['models_directory']).resolve()
        assert os.path.realpath(saved_config['output_directory']) == os.path.realpath(test_config['output_directory'])

def test_process_status_api(client):
    """Test process status API endpoints"""
    # Add a test process
    from civitai_manager.web_app import process_mgr
    process_id = process_mgr.add_process("test.safetensors")
    
    # Test getting single process status
    response = client.get(f'/api/process-status/{process_id}')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['filename'] == "test.safetensors"
    assert data['status'] == "pending"
    
    # Test getting all active processes
    response = client.get('/api/active-processes')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]['process_id'] == process_id
    
    # Test non-existent process
    response = client.get('/api/process-status/nonexistent')
    assert response.status_code == 404

def test_model_detail_page(client, test_config, mocker):
    """Test model detail page"""
    # Explicitly set app config for this test
    app.config['MODELS_DIR'] = test_config['models_directory']
    app.config['OUTPUT_DIR'] = test_config['output_directory']

    # Create a dummy model directory and files
    model_name = 'test_model'
    model_dir = Path(test_config['output_directory']) / model_name
    model_dir.mkdir()
    
    model_data = {
        'name': 'Test Model',
        'type': 'Checkpoint',
        'description': 'Test description',
        'modelVersions': [{
            'name': 'v1.0',
            'baseModel': 'SD 1.5'
        }]
    }
    with open(model_dir / f"{model_name}_civitai_model.json", 'w') as f:
        json.dump(model_data, f)
    with open(model_dir / f"{model_name}_civitai_model_version.json", 'w') as f:
        json.dump({"name": "v1.0", "baseModel": "SD 1.5", "images": []}, f)
    with open(model_dir / f"{model_name}_hash.json", 'w') as f:
        json.dump({"hash_value": "dummy_hash", "name": "test_model.safetensors"}, f)

    # Mock config validation and model data
    mocker.patch('civitai_manager.web_app.is_configured', return_value=True)
    mocker.patch('civitai_manager.web_app.load_web_config', return_value=test_config)
    
    mocker.patch('civitai_manager.web_app.load_model_data', return_value=model_data)
    mocker.patch('civitai_manager.web_app.find_model_file_path', return_value='test_model/test.safetensors')
    
    response = client.get(f'/model/{model_name}')
    assert response.status_code == 200
    assert b'Test Model' in response.data
    assert b'Test description' in response.data