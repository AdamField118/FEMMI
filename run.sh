#!/bin/bash
# Run tests/examples from project root
# This ensures imports work correctly

# Get script directory (should be project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Add project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run the requested script
if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script_name>"
    echo ""
    echo "Examples:"
    echo "  ./run.sh tests/example.py"
    echo "  ./run.sh tests/validation.py"
    echo ""
    echo "Or you can install the package:"
    echo "  pip install -e ."
    echo "  Then run directly: python tests/example.py"
    exit 1
fi

echo "Running: $1"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

python "$1"