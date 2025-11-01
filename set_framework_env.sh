#!/bin/bash

# Framework Environment Setup Script
# This script configures Python path and other environment variables for the framework

# Set current project path
if [[ -z "$CURR_PROJECT_PATH" ]]; then
    export CURR_PROJECT_PATH=$(pwd)
    echo "CURR_PROJECT_PATH was not set, setting it to current directory: $CURR_PROJECT_PATH"
else
    echo "CURR_PROJECT_PATH: $CURR_PROJECT_PATH"
fi

# Validate project path exists
if [[ ! -d "$CURR_PROJECT_PATH" ]]; then
    echo "Error: CURR_PROJECT_PATH directory does not exist: $CURR_PROJECT_PATH" >&2
    exit 1
fi

# Add project path to PYTHONPATH
export PYTHONPATH="${CURR_PROJECT_PATH}:$PYTHONPATH"
echo "Updated PYTHONPATH: $PYTHONPATH"

# Optional: Add additional library paths if needed
# Uncomment and modify the following line if you need specific library paths
# export LD_LIBRARY_PATH="/path/to/your/libs:$LD_LIBRARY_PATH"

# Verify Python can import the framework
if python3 -c "import framework" 2>/dev/null; then
    echo "Framework import test: SUCCESS"
else
    echo "Warning: Framework import test failed. Please check your Python environment and dependencies."
fi

echo "Environment setup completed successfully."