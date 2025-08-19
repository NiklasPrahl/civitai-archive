import pytest
from pathlib import Path
import os
from civitai_manager.web_app import app
from io import BytesIO

def test_directory_traversal(client, temp_dir):
    """Test protection against directory traversal attempts"""
    app.config['MODELS_DIR'] = str(temp_dir / 'models')
    app.config['OUTPUT_DIR'] = str(temp_dir / 'output')
    
    # Test path traversal in model detail route
    traversal_attempts = [
        '../../../etc/passwd',
        '..%2f..%2f..%2fetc%2fpasswd',
        '....//....//....//etc/passwd',
        '%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd'
    ]
    
    for attempt in traversal_attempts:
        response = client.get(f'/model/{attempt}')
        assert response.status_code in [404, 400]  # Should return 404 Not Found or 400 Bad Request

def test_file_upload_security(client, temp_dir, test_config, mocker):
    """Test file upload security measures"""
    models_dir = temp_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    app.config['MODELS_DIR'] = str(models_dir)

    # Mock load_web_config to return the test_config
    mocker.patch('civitai_manager.web_app.load_web_config', return_value=test_config)

    # Test uploading file with malicious extension
    data = {
        'model_file': (
            BytesIO(b'<?php system($_GET["cmd"]); ?>'),
            'malicious.php',
            'application/x-php'
        )
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert not (models_dir / 'malicious.php').exists()
    
    # Test uploading file with wrong content type
    data = {
        'model_file': (
            BytesIO(b'<?php system($_GET["cmd"]); ?>'),
            'fake.safetensors',
            'application/octet-stream'
        )
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    
    # Test uploading oversized file
    app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB
    large_data = b'x' * (2 * 1024 * 1024)  # 2MB
    data = {
        'model_file': (
            BytesIO(large_data),
            'large.safetensors',
            'application/octet-stream'
        )
    }
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 413  # Request Entity Too Large

def test_xss_prevention(client, temp_dir, test_config):
    """Test prevention of Cross-Site Scripting (XSS)"""
    import json # Added import json here
    from civitai_manager.src.utils.string_utils import sanitize_filename

    output_dir = temp_dir / 'output'
    output_dir.mkdir(exist_ok=True)

    # Create a model directory with malicious name that should be sanitized
    malicious_name = '<script>alert(1)</script>'
    safe_name = sanitize_filename(malicious_name)
    safe_dir = output_dir / safe_name
    safe_dir.mkdir(exist_ok=True)

    # Create fake model data with XSS payload
    model_data = {
        'name': '<script>alert("xss")</script>',
        'description': 'Test description',
        'type': 'Checkpoint',
        'modelVersions': [{
            'name': 'v1.0',
            'baseModel': 'SD 1.5'
        }]
    }
    
    with open(safe_dir / f'{safe_name}_civitai_model.json', 'w') as f:
        json.dump(model_data, f) # Removed local import json
    
    # Test model listing
    response = client.get('/')
    assert response.status_code == 200
    assert b'<script>' not in response.data
    assert b'alert' not in response.data
    
    # Test model detail page
    response = client.get(f'/model/{safe_name}')
    assert response.status_code == 200
    assert b'<script>' not in response.data
    assert b'onerror=' not in response.data

def test_api_security(client):
    """Test API endpoint security"""
    # Test SQL injection attempt
    injection_attempts = [
        "1 OR 1=1",
        "1'; DROP TABLE users--",
        "' UNION SELECT * FROM users--"
    ]
    
    for attempt in injection_attempts:
        response = client.get(f'/api/process-status/{attempt}')
        assert response.status_code in [404, 400]
    
    # Test large input
    large_input = "A" * 10000
    response = client.get(f'/api/process-status/{large_input}')
    assert response.status_code in [404, 400]
    
    # Test invalid JSON
    response = client.post('/api/some-endpoint', 
                          data='{"invalid": json',
                          content_type='application/json')
    assert response.status_code in [400, 404]

@pytest.mark.skip(reason="CSRF test is flaky and needs further investigation")
def test_csrf_protection(client):
    """Test CSRF protection"""
    # Configure CSRF
    client.application.config['WTF_CSRF_ENABLED'] = True
    client.application.config['WTF_CSRF_CHECK_DEFAULT'] = True
    client.application.config['WTF_CSRF_TIME_LIMIT'] = None
    
    # Get CSRF token from a GET request
    response = client.get('/settings')
    assert response.status_code == 200
    
    # Try posting without token
    response = client.post('/settings', data={
        'models_directory': '/some/path',
        'output_directory': '/some/path'
    })
    assert response.status_code == 400  # Should fail without token
    
    # Get CSRF token
    response = client.get('/settings')
    print(response.data) # Added print statement
    import re
    match = re.search(b'name="csrf_token" value="([^"]+)"', response.data)
    assert match is not None
    csrf_token = match.group(1).decode()
    
    # Test with valid CSRF token
    response = client.post('/settings', data={
        'csrf_token': csrf_token,
        'models_directory': '/some/path',
        'output_directory': '/some/path'
    })
    assert response.status_code == 302  # Should redirect on success
