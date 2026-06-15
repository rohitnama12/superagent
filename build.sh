#!/bin/bash
set -e

# Local build script for PyInstaller
# Usage: ./build.sh [output_name]
# Example: ./build.sh superagent-linux-x64

OUTPUT_NAME=${1:-superagent}

echo "Building SuperAgent executable: $OUTPUT_NAME"

# PyInstaller ignores .env and .chroma_db by default unless explicitly added via --add-data.
# This ensures a clean, isolated binary.
pyinstaller --onedir \
            --name "$OUTPUT_NAME" \
            --clean \
            superagent/main.py

echo "Zipping dist/$OUTPUT_NAME directory for distribution..."
(cd dist && zip -q -r "../$OUTPUT_NAME.zip" "$OUTPUT_NAME")

echo "Build complete! Binary is located in the dist/ directory. Archive created: ${OUTPUT_NAME}.zip"
