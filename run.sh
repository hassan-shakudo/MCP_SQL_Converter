#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Vanna Dremio NL-to-SQL Service..."
python main.py
