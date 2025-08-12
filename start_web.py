#!/usr/bin/env python3
"""
Start script for Civitai Manager web server
"""

import sys
import os

# Add project folder to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from civitai_manager.web_app import app
    
    if __name__ == "__main__":
        print("=" * 60)
        print("Civitai Manager Web Interface")
        print("=" * 60)
        print("Starting web application...")
        print("Open http://localhost:8080 in your browser")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=8080, debug=True)
        
except ImportError as e:
    print("Error importing web application:")
    print(f"  {e}")
    print("\nMake sure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("\nOr start the application via:")
    print("  python -m civitai_manager.main --web")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1) 