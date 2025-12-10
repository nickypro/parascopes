#!/bin/bash
# Script to run probe training for all Gemma models in sequence with the same training parameters

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Log file
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/all_probes_${TIMESTAMP}.log"

echo "============================================"
echo "   Gemma Probe Training Pipeline"
echo "============================================"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Function to run training for a specific model
run_probe() {
    local model=$1
    echo ""
    echo "========================================"
    echo "Starting training for: ${model}"
    echo "========================================"
    echo "Time: $(date)"
    
    export PARASCOPES_MODEL=$model
    
    echo "Config:"
    echo "  Model: $PARASCOPES_MODEL"
    echo ""
    
    # Run the probe training
    if python 03_probe_runner.py 2>&1 | tee -a "$LOG_FILE"; then
        echo ""
        echo "✓ Successfully completed training for ${model}"
        return 0
    else
        echo ""
        echo "✗ Failed training for ${model}"
        return 1
    fi
}

# Array of models to train
MODELS=("gemma4b" "gemma12b" "gemma27b")
FAILED_MODELS=()
SUCCESSFUL_MODELS=()

# Train each model in sequence
for model in "${MODELS[@]}"; do
    if run_probe "$model"; then
        SUCCESSFUL_MODELS+=("$model")
    else
        FAILED_MODELS+=("$model")
        echo "Error occurred with ${model}. Check log file: ${LOG_FILE}"
        
        # Ask if we should continue
        read -p "Continue with next model? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping pipeline."
            break
        fi
    fi
    
    # Add a separator between runs
    echo ""
    echo "========================================"
    echo "Completed: ${model}"
    echo "========================================"
    echo ""
done

# Final summary
echo ""
echo "============================================"
echo "   Training Pipeline Complete"
echo "============================================"
echo "Finished at: $(date)"
echo ""

if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    echo "Successful models (${#SUCCESSFUL_MODELS[@]}):"
    for model in "${SUCCESSFUL_MODELS[@]}"; do
        echo "  ✓ $model"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Failed models (${#FAILED_MODELS[@]}):"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model"
    done
    exit 1
fi

echo ""
echo "All models trained successfully!"
echo "Check wandb for results: https://wandb.ai/seperability/outlines_probes"
exit 0

