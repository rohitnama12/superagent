#!/bin/bash
set -e

# Local build script for PyInstaller
# Usage: ./build.sh [output_name]
# Example: ./build.sh superagent-linux-x64

OUTPUT_NAME=${1:-superagent}

echo "Building SuperAgent executable: $OUTPUT_NAME"

# PyInstaller ignores .env and .chroma_db by default unless explicitly added via --add-data.
# This ensures a clean, isolated binary.
pyinstaller --onefile \
            --name "$OUTPUT_NAME" \
            --clean \
            superagent/main.py

echo "Build complete! Binary is located in the dist/ directory."
