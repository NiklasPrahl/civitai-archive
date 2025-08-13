#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, BooleanField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import threading
import time

from civitai_manager.src.core.metadata_manager import (
    process_single_file,
    process_directory
)
from civitai_manager.src.utils.web_helpers import find_model_file_path, load_web_config, save_web_config
from civitai_manager.src.utils.html_generators.browser_page import generate_global_summary

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variable to hold the processing thread and a cancellation flag
processing_thread = None
cancel_processing_flag = threading.Event()

# Global configuration
CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json')) # Adjusted path
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'safetensors', 'ckpt', 'pt', 'pth', 'bin'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ConfigForm(FlaskForm):
    models_directory = StringField('Models Directory', validators=[DataRequired()])
    output_directory = StringField('Output Directory', validators=[DataRequired()])
    download_all_images = BooleanField('Download All Images')
    skip_images = BooleanField('Skip Images')
    notimeout = BooleanField('No Timeout (may trigger rate limiting)')
    submit = SubmitField('Save Configuration')

class UploadForm(FlaskForm):
    model_file = FileField('Model File', validators=[
        FileRequired(),
        FileAllowed(ALLOWED_EXTENSIONS, 'Only SafeTensors, CKPT, PT, PTH, and BIN files are allowed!')
    ])
    submit = SubmitField('Upload Model')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





def get_models_info():
    """Get information about all models for the dashboard."""
    config = load_web_config()
    output_dir = config.get('output_directory')
    models_dir = config.get('models_directory')

    if not output_dir or not os.path.exists(output_dir):
        return []

    summary_file_path = os.path.join(output_dir, 'all_models_summary.json')
    
    models = []
    # Try to load from the pre-generated summary file
    if os.path.exists(summary_file_path):
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                models = json.load(f)
            
            # Post-process models to add local preview image URLs
            for model in models:
                item = model['base_name'] # This is the sanitized name
                item_path = os.path.join(output_dir, item)
                local_images = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if local_images:
                    # Construct the relative path for local_static_files endpoint
                    model['preview_image_url'] = url_for('local_static_files', filename=f'{item}/{sorted(local_images)[0]}')
                    model['has_images'] = True
                else:
                    model['preview_image_url'] = url_for('static', filename='placeholder.png')
                    model['has_images'] = False

            return models
        except Exception:
            # Fallback to direct scan if summary file fails
            pass

    # Fallback to original, slower logic if summary file is not available or fails
    try:
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                model_info = {
                    'name': item,
                    'path': item_path,
                    'has_metadata': False,
                    'has_images': False,
                    'files': [],
                    'title': item,
                    'author': 'Unknown',
                    'tags': [],
                    'preview_image_url': url_for('static', filename='placeholder.png')
                }

                # Try to load metadata
                metadata = {}
                model_metadata_path = os.path.join(item_path, f'{item}_civitai_model.json')
                if os.path.exists(model_metadata_path):
                    with open(model_metadata_path, 'r') as f:
                        metadata = json.load(f)
                        model_info['has_metadata'] = True
                
                if metadata:
                    model_info['title'] = metadata.get('name', item)
                    if metadata.get('creator') and metadata['creator'].get('name'):
                        model_info['author'] = metadata['creator']['name']
                    model_info['tags'] = metadata.get('tags', [])

                # Robustly find local preview image
                local_images = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if local_images:
                    model_info['preview_image_url'] = url_for('local_static_files', filename=f'{item}/{sorted(local_images)[0]}')
                    model_info['has_images'] = True

                # Get model files from the models directory
                if models_dir and os.path.exists(models_dir):
                    model_hash_file = os.path.join(item_path, f'{item}_hash.json')
                    if os.path.exists(model_hash_file):
                        try:
                            with open(model_hash_file, 'r') as f:
                                hash_data = json.load(f)
                                stored_hash = hash_data.get('hash_value')
                                stored_filename = hash_data.get('filename')
                                if stored_hash and stored_filename:
                                    rel_path = find_model_file_path(models_dir, stored_hash, stored_filename)
                                    if rel_path:
                                        model_info['files'].append(rel_path)
                        except Exception as e:
                            print(f"Error finding model file for {item}: {e}")

                models.append(model_info)
    except Exception as e:
        print(f"Error reading models directory: {e}")

    return models

@app.route('/')
def index():
    """Main dashboard"""
    config = load_web_config()
    models = get_models_info()
    
    if not config.get('models_directory') or not config.get('output_directory'):
        return redirect(url_for('config'))
    
    return render_template('dashboard.html', config=config, models=models)

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Configuration page"""
    form = ConfigForm()
    current_config = load_web_config()
    
    if form.validate_on_submit():
        config_data = {
            'models_directory': form.models_directory.data,
            'output_directory': form.output_directory.data,
            'download_all_images': form.download_all_images.data,
            'skip_images': form.skip_images.data,
            'notimeout': form.notimeout.data
        }
        
        if save_web_config(config_data):
            flash('Configuration saved successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Error saving configuration!', 'error')
    
    if current_config:
        form.models_directory.data = current_config.get('models_directory', '')
        form.output_directory.data = current_config.get('output_directory', '')
        form.download_all_images.data = current_config.get('download_all_images', False)
        form.skip_images.data = current_config.get('skip_images', False)
        form.notimeout.data = current_config.get('notimeout', False)
    
    return render_template('config.html', form=form)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Model upload page"""
    form = UploadForm()
    config = load_web_config()
    
    if not config.get('models_directory') or not config.get('output_directory'):
        flash('Please configure directories first!', 'error')
        return redirect(url_for('config'))
    
    if form.validate_on_submit():
        if 'model_file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        file = request.files['model_file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)
            
            models_dir = config['models_directory']
            final_path = os.path.join(models_dir, filename)
            
            try:
                shutil.move(upload_path, final_path)
                
                def process_upload():
                    try:
                        output_dir = Path(config['output_directory'])
                        process_single_file(
                            Path(final_path), 
                            output_dir,
                            download_all_images=config.get('download_all_images', False),
                            skip_images=config.get('skip_images', False)
                        )
                        generate_global_summary(output_dir, models_dir)
                    except Exception as e:
                        print(f"Error processing uploaded file {filename}: {e}")
                
                thread = threading.Thread(target=process_upload)
                thread.daemon = True
                thread.start()
                
                flash(f'Model {filename} uploaded successfully! Processing started in the background.', 'success')
                return redirect(url_for('index'))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                if os.path.exists(upload_path):
                    os.remove(upload_path)
        else:
            flash('Invalid file type!', 'error')
    
    return render_template('upload.html', form=form)

@app.route('/process-all')
def process_all():
    """Process all models in the configured directory"""
    global processing_thread, cancel_processing_flag
    config = load_web_config()
    
    if not config.get('models_directory') or not config.get('output_directory'):
        return jsonify({'error': 'Configuration not set'}), 400
    
    def process_all_models_task(): # Renamed to avoid conflict with outer function
        global processing_thread, cancel_processing_flag
        cancel_processing_flag.clear() # Clear the flag at the start of a new process
        try:
            models_dir = Path(config['models_directory'])
            output_dir = Path(config['output_directory'])
            
            process_directory(
                models_dir,
                output_dir,
                config.get('notimeout', False),
                download_all_images=config.get('download_all_images', False),
                skip_images=config.get('skip_images', False),
                cancel_flag=cancel_processing_flag # Pass the flag
            )
            # Ensure thread is cleared and flag reset immediately after processing
            processing_thread = None
            cancel_processing_flag.clear()

            if not cancel_processing_flag.is_set(): # Only generate summary if not cancelled
                generate_global_summary(output_dir, models_dir)
        except Exception as e:
            print(f"Error processing all models: {e}")
        finally:
            processing_thread = None # Clear the thread reference when done
            cancel_processing_flag.clear() # Clear the flag
    
    if processing_thread and processing_thread.is_alive():
        return jsonify({'message': 'Processing already in progress'}), 409

    processing_thread = threading.Thread(target=process_all_models_task)
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({'message': 'Processing started'})

@app.route('/cancel-processing')
def cancel_processing():
    global processing_thread, cancel_processing_flag
    if processing_thread and processing_thread.is_alive():
        cancel_processing_flag.set() # Set the flag to signal cancellation
        return jsonify({'message': 'Cancellation requested'})
    else:
        return jsonify({'message': 'No active processing to cancel'}), 400


@app.route('/model/<model_name>')
def model_detail(model_name):
    """Show detailed information about a specific model"""
    start_time = time.time()
    print(f"DEBUG: Entering model_detail for {model_name} at {start_time}")

    config = load_web_config()
    output_dir = config.get('output_directory')

    if not output_dir:
        return redirect(url_for('config'))

    model_path = os.path.join(output_dir, model_name)
    if not os.path.exists(model_path):
        flash('Model not found!', 'error')
        return redirect(url_for('index'))

    metadata = {}
    version_data = {}
    model_files = []

    try:
        # Load model metadata
        load_metadata_start = time.time()
        model_metadata_path = os.path.join(model_path, f'{model_name}_civitai_model.json')
        if os.path.exists(model_metadata_path):
            try:
                with open(model_metadata_path, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: JSONDecodeError when loading model metadata from {model_metadata_path}: {e}")
                metadata = {} # Ensure metadata is empty on error
            except Exception as e:
                print(f"ERROR: Unexpected error when loading model metadata from {model_metadata_path}: {e}")
                metadata = {} # Ensure metadata is empty on error
        print(f"DEBUG: Loaded model metadata in {time.time() - load_metadata_start:.4f} seconds")

        # Load version metadata
        load_version_start = time.time()
        version_metadata_path = os.path.join(model_path, f'{model_name}_civitai_model_version.json')
        if os.path.exists(version_metadata_path):
            try:
                with open(version_metadata_path, 'r') as f:
                    version_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: JSONDecodeError when loading version metadata from {version_metadata_path}: {e}")
                version_data = {} # Ensure version_data is empty on error
            except Exception as e:
                print(f"ERROR: Unexpected error when loading version metadata from {version_metadata_path}: {e}")
                version_data = {} # Ensure version_data is empty on error
        print(f"DEBUG: Loaded version metadata in {time.time() - load_version_start:.4f} seconds")
        
        if metadata and 'modelVersions' in metadata and version_data:
            version_id = version_data.get('id')
            for v in metadata.get('modelVersions', []):
                if v.get('id') == version_id:
                    temp_version = v.copy()
                    temp_version.update(version_data)
                    version_data = temp_version
                    break

        # Process images
        process_images_start = time.time()
        if version_data.get('images'):
            for i, image_data in enumerate(version_data['images']):
                if image_data.get('type') == 'video':
                    ext = '.mp4'
                else:
                    url = image_data.get('url', '')
                    ext = Path(url.split('?')[0]).suffix if url else '.jpeg'

                image_filename = f"{model_name}/{model_name}_preview_{i}{ext}"
                full_image_path = os.path.join(output_dir, image_filename)
                
                if os.path.exists(full_image_path):
                    version_data['images'][i]['local_url'] = url_for('local_static_files', filename=image_filename)
                else:
                    version_data['images'][i]['local_url'] = url_for('static', filename='placeholder.png')
        print(f"DEBUG: Processed images in {time.time() - process_images_start:.4f} seconds")

        # Fallback for metadata
        if not metadata:
            fallback_metadata_start = time.time()
            model_info_path = os.path.join(model_path, 'model_info.json')
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    metadata = json.load(f)
            print(f"DEBUG: Loaded fallback metadata in {time.time() - fallback_metadata_start:.4f} seconds")

        # Find model files
        find_files_start = time.time()
        if version_data:
            models_dir = config.get('models_directory')
            file_info = version_data.get('files', [])
            if models_dir and os.path.exists(models_dir) and file_info:
                stored_hash = file_info[0].get('hashes', {}).get('SHA256')
                stored_filename = file_info[0].get('name') # This is the original filename, e.g., 'flux_dev.safetensors'

                if stored_hash and stored_filename:
                    expected_path = os.path.join(models_dir, stored_filename)
                    if os.path.exists(expected_path): # Only check for existence, trust the hash from _hash.json
                        model_files.append(os.path.relpath(expected_path, models_dir))
                    # No fallback to find_model_file_path here, as it's slow and we have the filename
        print(f"DEBUG: Found model files in {time.time() - find_files_start:.4f} seconds")

    except Exception as e:
        print(f"Error reading model details for {model_name}: {e}")

    print(f"DEBUG: Rendering model_detail.html for {model_name}. Total time: {time.time() - start_time:.4f} seconds")

    likes_fill_width = 0
    if version_data and version_data.get('stats'):
        thumbs_up = version_data['stats'].get('thumbsUpCount', 0)
        thumbs_down = version_data['stats'].get('thumbsDownCount', 0)
        total_votes = thumbs_up + thumbs_down
        if total_votes > 0:
            likes_fill_width = (thumbs_up / total_votes) * 100

    return render_template('model_detail.html',
                         model_name=model_name,
                         model_files=model_files,
                         metadata=metadata,
                         version=version_data,
                         likes_fill_width=likes_fill_width)

@app.route('/local_static/<path:filename>')
def local_static_files(filename):
    """Serve static files from the output directory"""
    config = load_web_config()
    output_dir = config.get('output_directory')

    if not output_dir:
        return '', 404

    try:
        return send_from_directory(output_dir, filename)
    except Exception as e:
        print(f"DEBUG: Error serving file {filename} from {output_dir}: {e}")
        return '', 404

@app.route('/api/models')
def api_models():
    """API endpoint to get models information"""
    models = get_models_info()
    return jsonify(models)

@app.route('/api/status')
def api_status():
    """API endpoint to get processing status"""
    config = load_web_config()
    return jsonify({
        'configured': bool(config.get('models_directory') and config.get('output_directory')),
        'models_directory': config.get('models_directory', ''),
        'output_directory': config.get('output_directory', '')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)