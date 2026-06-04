#!/usr/bin/env bash
set -euo pipefail

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python main.py"
