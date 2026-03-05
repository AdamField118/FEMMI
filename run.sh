#!/bin/bash
# Run scripts from the FEMMI project root.
# This ensures femmi/ package is importable without installing.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script>"
    echo ""
    echo "Tests:"
    echo "  ./run.sh tests/test_pipeline.py"
    echo "  ./run.sh tests/test_convergence_p3.py"
    echo ""
    echo "Examples:"
    echo "  ./run.sh examples/demo_p3_pipeline.py"
    echo "  ./run.sh examples/cluster_example.py"
    echo ""
    echo "Or install in development mode and run directly:"
    echo "  pip install -e ."
    exit 1
fi

echo "Running: $1"
echo "PYTHONPATH: $PYTHONPATH"
echo ""
python "$1"