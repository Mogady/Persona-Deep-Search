#!/bin/bash

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Chainlit
chainlit run src/ui/chainlit_app.py --host 0.0.0.0 --port 8000
