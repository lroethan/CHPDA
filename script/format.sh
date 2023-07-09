#!/bin/bash

if ! command -v black &> /dev/null; then
    echo "Error: Black is not installed. Please install Black (https://github.com/psf/black) before running this script."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Error: No file or directory specified."
    echo "Usage: ./format.sh <file or directory>"
    exit 1
fi

target="$1"

if [ -f "$target" ]; then
    black "$target"
    echo "Formatted file: $target"
elif [ -d "$target" ]; then
    find "$target" -name "*.py" -exec black --line-length 140 {} \;
    echo "Formatted files in directory: $target"
else
    echo "Error: Invalid file or directory specified."
    echo "Usage: ./format.sh <file or directory>"
    exit 1
fi