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
from typing import Dict, Optional
from datetime import datetime
import html # Add this import

from civitai_manager.src.core.metadata_manager import (
    process_single_file,
    process_directory
)
from civitai_manager.src.utils.web_helpers import find_model_file_path, load_web_config, save_web_config
from civitai_manager.src.utils.html_generators.browser_page import generate_global_summary
from civitai_manager.src.utils.file_tracker import ProcessedFilesManager
from civitai_manager.src.utils.process_manager import ProcessManager
from civitai_manager.src.utils.string_utils import sanitize_filename

# Initialize process manager
process_mgr = ProcessManager()

# Flask app configuration
ALLOWED_EXTENSIONS = {'safetensors', 'ckpt', 'pt', 'pth', 'bin'}
ALLOWED_MIMETYPES = ['application/octet-stream']
ALLOWED_MIMETYPES = ['application/octet-stream']
ALLOWED_MIMETYPES = ['application/octet-stream']
ALLOWED_MIMETYPES = ['application/octet-stream']
ALLOWED_MIMETYPES = ['application/octet-stream']
ALLOWED_MIMETYPES = ['application/octet-stream']
CONFIG_FILE = os.environ.get('CONFIG_FILE', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json')))
MODELS_DIR = os.environ.get('MODELS_DIR', '')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['CONFIG_FILE'] = CONFIG_FILE

def create_app():
    from flask_wtf.csrf import CSRFProtect
    csrf = CSRFProtect(app)

    # Load config to get SECRET_KEY
    # TODO: For production, SECRET_KEY should be loaded from environment variables or a secure secret management system.
    current_config = load_web_config(app.config['CONFIG_FILE'])
    app.config['SECRET_KEY'] = current_config.get('SECRET_KEY', os.urandom(24).hex())

    return app

app = create_app()

processing_thread = None
cancel_processing_flag = threading.Event()

def is_configured(app_instance):
    """Check if the application is configured with valid directories"""
    models_dir = app_instance.config.get('models_directory') # Use 'models_directory' from config
    output_dir = app_instance.config.get('output_directory') # Use 'output_directory' from config
    
    print(f"DEBUG: is_configured check - models_dir: {models_dir}, output_dir: {output_dir}")
    
    models_dir_exists = False
    if models_dir:
        models_dir_path = Path(models_dir)
        models_dir_exists = models_dir_path.exists()
        print(f"DEBUG: models_dir_path: {models_dir_path}, exists: {models_dir_exists}")
    
    output_dir_exists = False
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_exists = output_dir_path.exists()
        print(f"DEBUG: output_dir_path: {output_dir_path}, exists: {output_dir_exists}")
        
    return (models_dir and output_dir and models_dir_exists and output_dir_exists)

# Initialize default config if not exists
if not os.path.exists(CONFIG_FILE):
    default_config = {
        'models_directory': MODELS_DIR,
        'output_directory': OUTPUT_DIR,
        'download_all_images': False,
        'skip_images': False,
        'notimeout': False
    }
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(default_config, f, indent=4)

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
                                FileAllowed(list(ALLOWED_EXTENSIONS) + ALLOWED_MIMETYPES, 'Only SafeTensors, CKPT, PT, PTH, BIN files and octet-stream mimetype are allowed!')
    ])
    submit = SubmitField('Upload Model')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_data(model_file, output_dir):
    """Load the model data from the metadata files"""
    model_data = {
        'name': 'Unknown',
        'type': 'Unknown',
        'description': 'No description available',
        'version': {
            'name': 'Unknown',
            'baseModel': 'Unknown'
        }
    }
    
    # Try to load model metadata
    model_name = Path(model_file).stem
    model_dir = Path(output_dir) / model_name
    
    # Load version data
    version_file = model_dir / f"{model_name}_civitai_model_version.json"
    if version_file.exists():
        try:
            with open(version_file, 'r') as f:
                model_data['version'] = json.load(f)
        except Exception as e:
            print(f"Error loading version data: {e}")
            
    # Load model data
    model_file = model_dir / f"{model_name}_civitai_model.json"
    if model_file.exists():
        try:
            with open(model_file, 'r') as f:
                data = json.load(f)
                model_data.update(data)
        except Exception as e:
            print(f"Error loading model data: {e}")
            
    return model_data



def get_models_info():
    """Get information about all models for the dashboard."""
    start_time = time.time()
    print("DEBUG: get_models_info started.")
    config = load_web_config(app.config['CONFIG_FILE'])
    output_dir = config.get('output_directory')
    models_dir = config.get('models_directory')

    if not output_dir or not os.path.exists(output_dir):
        print("DEBUG: Output directory not configured or does not exist.")
        return []

    summary_file_path = os.path.join(output_dir, 'all_models_summary.json')
    
    models = []
    # Try to load from the pre-generated summary file
    if os.path.exists(summary_file_path):
        print(f"DEBUG: Found summary file: {summary_file_path}")
        try:
            with open(summary_file_path, 'r', encoding='utf-8') as f:
                models = json.load(f)
            print(f"DEBUG: Successfully loaded models from summary file. Count: {len(models)}")
            
            # Post-process models to add local preview image URLs and escape HTML
            for model in models:
                item = model['base_name'] # This is the sanitized name
                # Ensure base_name is also HTML escaped for URL generation
                model['base_name'] = html.escape(item) # Add this line
                item_path = os.path.join(output_dir, item)
                local_images = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if local_images:
                    # Construct the relative path for local_static_files endpoint
                    model['preview_image_url'] = url_for('local_static_files', filename=f'{item}/{sorted(local_images)[0]}')
                    model['has_images'] = True
                else:
                    model['preview_image_url'] = url_for('static', filename='placeholder.png')
                    model['has_images'] = False
                
                # Explicitly escape HTML for title and author
                model['title'] = html.escape(model.get('title', ''))
                model['author'] = html.escape(model.get('author', 'Unknown'))
                model['tags'] = [html.escape(tag) for tag in model.get('tags', [])]

            print(f"DEBUG: Finished post-processing models from summary file. Total time: {time.time() - start_time:.4f} seconds")
            return models
        except Exception as e:
            print(f"ERROR: Failed to load or parse summary file: {e}. Falling back to direct scan.")
            # Fallback to direct scan if summary file fails
            pass

    print("DEBUG: Performing direct scan of output directory (slower).")
    # Fallback to original, slower logic if summary file is not available or fails
    try:
        for item in os.listdir(output_dir):
            # Exclude the _bin directory from being processed as a model
            if item == '_bin':
                continue
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                model_info = {
                    'name': html.escape(item),
                    'path': item_path,
                    'has_metadata': False,
                    'has_images': False,
                    'files': [],
                    'title': html.escape(item),
                    'author': 'Unknown',
                    'tags': [],
                    'preview_image_url': url_for('static', filename='placeholder.png')
                }

                # Try to load metadata
                metadata = {}
                model_metadata_path = os.path.join(item_path, f'{item}_civitai_model.json')
                if os.path.exists(model_metadata_path):
                    try:
                        with open(model_metadata_path, 'r') as f:
                            metadata = json.load(f)
                            model_info['has_metadata'] = True
                    except Exception as e:
                        print(f"ERROR: Could not load metadata for {item}: {e}")
                
                if metadata:
                    model_info['title'] = metadata.get('name', item)
                    # Explicitly escape HTML for title and author
                    model_info['title'] = html.escape(model_info['title'])
                    if metadata.get('creator') and metadata['creator'].get('name'):
                        model_info['author'] = html.escape(metadata['creator']['name'])
                    model_info['tags'] = [html.escape(tag) for tag in metadata.get('tags', [])]

                # Robustly find local preview image
                local_images = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if local_images:
                    model_info['preview_image_url'] = url_for('local_static_files', filename=f'{item}/{sorted(local_images)[0]}')
                    model_info['has_images'] = True
                else:
                    model_info['preview_image_url'] = url_for('static', filename='placeholder.png')
                    model_info['has_images'] = False

                # Get model files from the models directory
                if models_dir and os.path.exists(models_dir):
                    model_hash_file = os.path.join(item_path, f'{item}_hash.json')
                    if os.path.exists(model_hash_file):
                        try:
                            with open(model_hash_file, 'r') as f:
                                hash_data = json.load(f)
                                stored_hash = hash_data.get('hash_value')
                                stored_filename = hash_data.get('name') # This is the original filename, e.g., 'flux_dev.safetensors'

                                if stored_hash and stored_filename:
                                    rel_path = find_model_file_path(models_dir, stored_hash, stored_filename)
                                    if rel_path:
                                        model_info['files'].append(rel_path)
                        except Exception as e:
                            print(f"ERROR: Error finding model file for {item}: {e}")

                models.append(model_info)
    except Exception as e:
        print(f"ERROR: Error reading models directory during direct scan: {e}")

    print(f"DEBUG: Finished direct scan. Total time: {time.time() - start_time:.4f} seconds")
    return models

@app.route('/api/process-status/<process_id>')
def get_process_status(process_id):
    """Get the status of a processing task"""
    status = process_mgr.get_status(process_id)
    if status:
        return jsonify({
            'status': status.status,
            'filename': status.filename,
            'progress': status.progress,
            'error': status.error,
            'start_time': status.start_time.isoformat(),
            'end_time': status.end_time.isoformat() if status.end_time else None
        })
    return jsonify({'error': 'Process not found'}), 404

@app.route('/api/active-processes')
def get_active_processes():
    """Get all active processing tasks"""
    active = process_mgr.get_all_active()
    return jsonify([{
        'process_id': p.filename,
        'status': p.status,
        'filename': p.filename,
        'progress': p.progress,
        'start_time': p.start_time.isoformat()
    } for p in active])

@app.route('/')
@app.route('/page/<int:page>')
def index(page=1):
    """Main dashboard with pagination"""
    config = load_web_config(app.config['CONFIG_FILE'])
    app.config.update(config) # Update app.config with loaded values
    
    if not config or not config.get('models_directory') or not config.get('output_directory'):
        return redirect(url_for('settings'))
    
    # Check if directories exist
    if not is_configured(app):
        return redirect(url_for('settings'))
    
    models = get_models_info()
    per_page = 20  # Number of models per page
    total = len(models)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_models = models[start_idx:end_idx]
    total_pages = (total + per_page - 1) // per_page
    
    return render_template('dashboard.html', 
                         config=config,
                         models=paginated_models,
                         total_models=total,
                         current_page=page,
                         total_pages=total_pages)

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Settings page"""
    form = ConfigForm()
    current_config = load_web_config(app.config['CONFIG_FILE'])
    app.config.update(current_config) # Update app.config with loaded values
    
    if form.validate_on_submit():
        config_data = {
            'models_directory': form.models_directory.data,
            'output_directory': form.output_directory.data,
            'download_all_images': form.download_all_images.data,
            'skip_images': form.skip_images.data,
            'notimeout': form.notimeout.data
        }
        
        if save_web_config(config_data, app.config['CONFIG_FILE']):
            flash('Settings saved successfully!', 'success')
            app.config.update(config_data) # Update app.config immediately after saving
            return redirect(url_for('settings'))
        else:
            flash('Error saving settings!', 'error')
    
    if current_config:
        form.models_directory.data = current_config.get('models_directory', '')
        form.output_directory.data = current_config.get('output_directory', '')
        form.download_all_images.data = current_config.get('download_all_images', True)
        form.skip_images.data = current_config.get('skip_images', False)
        form.notimeout.data = current_config.get('notimeout', False)
    
    return render_template('settings.html', form=form)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Model upload page"""
    global processing_thread
    form = UploadForm()
    config = load_web_config(app.config['CONFIG_FILE'])
    
    if not is_configured(app):
        flash('Please configure directories first!', 'error')
        return redirect(url_for('settings'))
    
    if form.validate_on_submit():
        print(f"DEBUG: Starting upload process for {form.model_file.data.filename}")
        if processing_thread and processing_thread.is_alive():
            print("DEBUG: Processing thread already active, returning 409")
            return jsonify({'success': False, 'message': 'A process is already running.'}), 409

        file = form.model_file.data
        filename = secure_filename(file.filename)
        
        models_dir = config['models_directory']
        final_path = os.path.join(models_dir, filename)

        # Save the file directly to the models directory
        print(f"DEBUG: Saving file to {final_path}")
        file.save(final_path)
        
        # Add process to manager and queue processing
        process_id = process_mgr.add_process(filename)
        
        def process_upload(process_id: str, file_path: str, output_dir: Path, models_dir: Path, config: Dict):
            try:
                process_mgr.update_status(process_id, 'processing', progress=0.0)
                
                # Process the file
                success = process_single_file(
                    Path(file_path), 
                    output_dir,
                    download_all_images=config.get('download_all_images', True),
                    skip_images=config.get('skip_images', False)
                )
                
                if success:
                    process_mgr.update_status(process_id, 'processing', progress=0.5)
                    # Update global summary
                    generate_global_summary(Path(output_dir), Path(models_dir))
                    process_mgr.update_status(process_id, 'completed', progress=1.0)
                else:
                    process_mgr.update_status(process_id, 'failed', 
                                           error="File processing failed")
                    
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                process_mgr.update_status(process_id, 'failed', error=error_msg)
                logging.error(f"Error processing {filename}: {error_msg}")

        # Queue the processing
        process_mgr.queue_process(
            process_upload,
            process_id,
            final_path,
            Path(config['output_directory']),
            Path(models_dir),
            config
        )
        
        return jsonify({
            'success': True, 
            'message': f'Model {filename} uploaded successfully! Processing started in the background.',
            'process_id': process_id
        })

    if request.method == 'POST':
        # Handle form validation errors for AJAX request
        errors = {field: error[0] for field, error in form.errors.items()}
        return jsonify({'success': False, 'message': 'There was an error with your submission.', 'errors': errors}), 400

    return render_template('upload.html', form=form)

@app.route('/process-all')
def process_all():
    """Process all models in the configured directory"""
    global processing_thread, cancel_processing_flag
    config = load_web_config(app.config['CONFIG_FILE'])
    
    if not is_configured(app):
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
                download_all_images=config.get('download_all_images', True),
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

    config = load_web_config(app.config['CONFIG_FILE'])
    output_dir = config.get('output_directory')
    output_dir = config.get('output_directory')

    if not is_configured(app):
        return redirect(url_for('settings'))

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
    config = load_web_config(app.config['CONFIG_FILE'])
    output_dir = config.get('output_directory')

    if not output_dir:
        return '', 404

    try:
        return send_from_directory(output_dir, filename)
    except Exception as e:
        print(f"DEBUG: Error serving file {filename} from {output_dir}: {e}")
        return '', 404

@app.route('/model/<model_name>/delete', methods=['POST'])
def delete_model(model_name):
    """Delete a model and its associated files."""
    config = load_web_config(app.config['CONFIG_FILE'])
    output_dir = Path(config.get('output_directory'))
    models_dir = Path(config.get('models_directory'))
    
    delete_model_file = request.form.get('delete_model_file') == 'true'
    
    model_output_path = output_dir / model_name
    
    if model_output_path.exists():
        original_filename = None
        hash_file_path = model_output_path / f'{model_name}_hash.json'
        if hash_file_path.exists():
            with open(hash_file_path, 'r') as f:
                hash_data = json.load(f)
                original_filename = hash_data.get('filename')

        bin_dir = output_dir / '_bin'
        bin_dir.mkdir(exist_ok=True)
        
        message = f'Model data for {model_name}'
        if delete_model_file and models_dir.exists() and original_filename:
            model_file_path = models_dir / original_filename
            if model_file_path.exists():
                shutil.move(str(model_file_path), str(bin_dir / original_filename))
                message = f'Model {original_filename} and its data'

        # Check if destination for model output path already exists in bin and remove it
        dest_model_output_in_bin = bin_dir / model_name
        if dest_model_output_in_bin.exists():
            if dest_model_output_in_bin.is_dir():
                shutil.rmtree(dest_model_output_in_bin)
            else:
                os.remove(dest_model_output_in_bin)
        
        shutil.move(str(model_output_path), str(bin_dir))
        flash(f'{message} moved to bin.', 'success')
        
        if original_filename:
            manager = ProcessedFilesManager(output_dir)
            manager.remove_processed_file(models_dir / original_filename)
            manager.save_processed_files()

        generate_global_summary(output_dir, models_dir)

    else:
        flash(f'Model {model_name} not found in output directory.', 'error')
        
    return redirect(url_for('index'))

@app.route('/settings/clear_bin', methods=['POST'])
def clear_bin():
    config = load_web_config(app.config['CONFIG_FILE'])
    output_dir = Path(config.get('output_directory'))
    bin_dir = output_dir / '_bin'
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
        flash('Deleted model bin cleared.', 'success')
    else:
        flash('Bin is already empty.', 'info')
    return redirect(url_for('settings'))



@app.route('/api/models')
def api_models():
    """API endpoint to get models information"""
    models = get_models_info()
    return jsonify(models)

@app.route('/api/status')
def api_status():
    """API endpoint to get processing status"""
    global processing_thread
    config = load_web_config(app.config['CONFIG_FILE'])
    return jsonify({
        'configured': is_configured(app),
        'models_directory': config.get('models_directory', ''),
        'output_directory': config.get('output_directory', ''),
        'is_processing': processing_thread and processing_thread.is_alive()
    })

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)