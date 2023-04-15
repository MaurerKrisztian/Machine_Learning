#!/bin/bash

# Install PyInstaller if not installed
if ! command -v pyinstaller &> /dev/null
then
    pip install pyinstaller
fi

# Remove old build and dist directories if they exist
rm -rf build/ dist/

# Build the executable
if pyinstaller --onefile --add-data "model.pkl:." cli.py; then
    # Copy the model file to the dist directory
    if cp ../model/my_model.pkl dist/model.pkl; then
        # Delete the build directory and the cli.spec file
        rm -rf build/ cli.spec
        echo "Executable built successfully in the dist directory."
    else
        echo "Error: failed to copy model file to dist directory."
    fi
else
    echo "Error: failed to build executable with PyInstaller."
fi
