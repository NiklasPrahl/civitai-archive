# Civitai Data Manager

A lightweight tool for local management, backup, and organization of SafeTensors model metadata from Civitai.

## Features

- **Web Interface**: Modern web interface for easy management
- **Automatic Metadata Detection**: Downloads metadata from Civitai based on file hashes
- **Preview Images**: Automatically downloads preview images for all models
- **HTML Overviews**: Generates searchable HTML overviews of all models
- **Batch Processing**: Processes entire directories at once
- **Upload Functionality**: Upload new models directly through the web interface

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
```

## Usage

### Web Interface (Recommended)

1. **Start the web application:**
   ```bash
   python start_web.py
   ```
   
   Or via the main module:
   ```bash
   python -m civitai_manager.main --web
   ```

2. **Open in browser:**
   ```
   http://localhost:8080
   ```

3. **Configure directories:**
   - Models Directory: Where your .safetensors, .ckpt, .pt, .pth, .bin files are located
   - Output Directory: Where metadata, images and HTML files will be stored

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
├── web_app.py              # Flask web application
├── main.py                 # CLI main program
├── templates/              # HTML templates
│   ├── base.html          # Base template
│   ├── dashboard.html     # Dashboard
│   ├── config.html        # Configuration
│   ├── upload.html        # Upload form
│   └── model_detail.html  # Model details
└── src/                   # Core functionality
    ├── core/              # Metadata processing
    ├── utils/             # Utility functions
    └── html_generators/   # HTML generation
```

### Start Web Application

```bash
# Development
python start_web.py

# Production
python -m civitai_manager.main --web --host 0.0.0.0 --port 8080
```

## Troubleshooting

### Port 5000 Already in Use (macOS)

On macOS, port 5000 is often used by AirPlay Receiver. The application now uses port 8080 by default. If you need to use a different port:

```bash
python -m civitai_manager.main --web --port 9000
```

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features.