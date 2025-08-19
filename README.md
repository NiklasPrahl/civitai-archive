# Civitai Data Manager

A lightweight web ui for local management, backup, and organization of SafeTensors model metadata from Civitai.

IMPORTANT: This project is currently in early development!

## Features

- **Web Interface**: Modern web interface for easy management
- **Automatic Metadata Detection**: Downloads metadata from Civitai based on file hashes
- **Preview Images**: Automatically downloads preview images for all models
- **HTML Overviews**: Generates searchable HTML overviews of all models
- **Batch Processing**: Processes entire directories at once
- **Upload Functionality**: Upload new models directly through the web interface
- **Docker Support**: Run as a container or TrueNAS Scale app
- **TrueNAS Integration**: Full support for TrueNAS Scale deployment

## Installation Options

### 1. Standard Installation (Local Development)

#### Install Dependencies
```bash
pip install -r requirements.txt
```

Or with Poetry:
```bash
poetry install
```

#### Start the Web Interface
```bash
python start_web.py
```

Or via the main module:
```bash
python -m civitai_manager.main --web
```

### 2. Docker Installation

#### Using Docker Compose (Development)
1. Clone the repository:
   ```bash
   git clone https://github.com/NiklasPrahl/civitai-archive.git
   cd civitai-archive
   ```

2. Edit docker-compose.yml to configure your paths:
   ```yaml
   volumes:
     - /path/to/models:/data/models:ro
     - civitai_output:/data/output
   ```

3. Start the container:
   ```bash
   docker-compose up -d
   ```

### 3. TrueNAS Scale Installation

#### Prerequisites
- TrueNAS Scale installed and configured
- Access to the TrueNAS Scale web interface
- Storage pool configured for your models

#### Option A: Direct Docker Installation on TrueNAS
1. Prepare directories on TrueNAS:
   ```bash
   mkdir -p /mnt/pool/models
   mkdir -p /mnt/pool/civitai-output
   ```

2. Set permissions:
   ```bash
   chown -R 1000:1000 /mnt/pool/civitai-output
   chmod -R 755 /mnt/pool/models
   ```

3. Deploy container:
   ```bash
   docker run -d \
     --name civitai-manager \
     -p 5000:5000 \
     -v /mnt/pool/models:/data/models:ro \
     -v /mnt/pool/civitai-output:/data/output \
     civitai-manager
   ```

#### Option B: As TrueNAS Scale Custom App (Recommended)
1. Create a custom app:
   - Go to "Apps" -> "Custom Apps"
   - Click "Upload Custom App" (the cloud icon with up arrow)
   - Choose either:
     - Option 1: Upload the pre-built `civitai-manager.tgz` from releases
     - Option 2: Build and upload your own chart:
       ```bash
       cd helm/civitai-manager
       helm package .
       ```

2. Install the Custom App:
   - After uploading, the app will appear in your Custom Apps list
   - Click "Install"
   - Configure in the installation wizard:
     - Models Path: Choose your dataset, e.g., `/mnt/pool/models` (This is where your existing model files are stored)
     - App Name: e.g., `civitai-manager`
     - Namespace: e.g., `ix-civitai-manager`

   Note on Storage:
   - Output Storage is used for permanent storage of metadata, preview images, generated HTML files, and temporary upload processing
   - The actual model files are read directly from your Models Path and don't need additional storage
   - Click "Save" and then "Install"

Note: This installation method keeps the app private to your TrueNAS instance and doesn't require adding external catalogs.

4. **Upload or process models:**
   - Use "Model Upload" to add new models
   - Use "Process All Models" to update existing models

### Command Line (CLI)

#### Process a single file

```bash
python -m civitai_manager.main --single /path/to/model.safetensors --output /path/to/output
```

#### Process all files in a directory

```bash
python -m civitai_manager.main --all /path/to/models/directory --output /path/to/output
```

#### Additional options

```bash
python -m civitai_manager.main --all /path/to/models --output /path/to/output --images --notimeout
```

- `--images`: Download all available preview images
- `--noimages`: Skip downloading images
- `--notimeout`: No timeout between files (may trigger rate limiting)
- `--onlynew`: Only process new files
- `--onlyupdate`: Only update existing files
- `--clean`: Remove data for models that no longer exist

## Configuration

### Web Interface (Recommended)

All configuration is now done through the web interface under "Configuration":

- **Models Directory**: Directory with your model files
- **Output Directory**: Directory for metadata and generated files
- **Processing Options**: Download images, timeouts, etc.

### Configuration File

The `config.json` file is automatically managed by the web interface. You can:

1. **Start the web application**: `python start_web.py`
2. **Configure directories** through the web interface
3. **All settings are automatically saved** to `config.json`

### Legacy CLI Configuration

If you need to use command line arguments instead of the web interface:

```bash
# Process files with explicit paths
python -m civitai_manager.main --all /path/to/models --output /path/to/output --images

# Use --noconfig to ignore any existing config.json
python -m civitai_manager.main --all /path/to/models --output /path/to/output --noconfig
```

**Note**: The web interface is now the primary way to configure the application. The `config.json` file is automatically created and managed when you use the web interface.

## Supported File Formats

- `.safetensors` - Safe tensor files
- `.ckpt` - Checkpoint files
- `.pt` / `.pth` - PyTorch models
- `.bin` - Binary model files

## Web Interface Features

### Dashboard
- Overview of all models
- Status of metadata and images
- Quick access to all functions

### Model Upload
- Drag & drop upload
- Progress display
- Automatic processing after upload
- Support for large files (up to 500MB)

### Model Management
- Detailed model information
- Display preview images
- Export metadata
- Direct links to Civitai

### Configuration
- Simple path input
- Processing options
- Input validation

## Development

### Project Structure

```
civitai_manager/
├── config.json                # Main config (managed by web UI)
├── main.py                    # CLI entry point
├── web_app.py                 # Flask web application
├── templates/                 # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── model_detail.html
│   ├── settings.html
│   ├── upload.html
├── src/
│   ├── core/                  # Metadata processing
│   ├── migrations/            # Migration scripts
│   ├── utils/                 # Utility functions
```

### Start Web Application

```bash
# Development
python start_web.py

# Production
python -m civitai_manager.main --web --host 0.0.0.0 --port 8080
```


```bash
python -m civitai_manager.main --web --port 9000
```

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features.