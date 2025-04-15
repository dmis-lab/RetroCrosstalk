#!/bin/bash
#
# Parallel EG_MCTS Job Execution Script
# 
# This script divides a range of values into multiple splits and runs the EG_MCTS.py
# script in parallel for each split, distributing the workload across processes.
# It's designed to process a dataset in chunks using a specified single-step model.
#

# ==================== Configuration ====================

# Set GPU to use
export CUDA_VISIBLE_DEVICES=4

# Model and dataset configuration
SS_MODEL="ReactionT5"
DATASET="literature30"

# Range configuration
START_VALUE=0
END_VALUE=30
NUM_SPLITS=5  # Divide the work into this many chunks

# ==================== Job Distribution Logic ====================

# Calculate the step size based on number of splits
STEP_SIZE=$(( (END_VALUE - START_VALUE + NUM_SPLITS - 1) / NUM_SPLITS ))

echo "========================================================"
echo "Starting parallel execution with the following settings:"
echo "  - Single-step model: $SS_MODEL"
echo "  - Dataset: $DATASET"
echo "  - Range: $START_VALUE to $END_VALUE"
echo "  - Number of splits: $NUM_SPLITS"
echo "  - Step size: $STEP_SIZE"
echo "  - Using GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# Run the Python script multiple times with different start and end arguments
CURRENT_START=$START_VALUE
while [ $CURRENT_START -lt $END_VALUE ]; do
    # Calculate the end for this split
    CURRENT_END=$((CURRENT_START + STEP_SIZE))
    
    # Ensure we don't exceed the overall end value
    if [ $CURRENT_END -gt $END_VALUE ]; then
        CURRENT_END=$END_VALUE
    fi
    
    # Run the Python script in the background with the current range
    echo "Starting job: python ./EG_MCTS.py --start $CURRENT_START --end $CURRENT_END --ss_model $SS_MODEL --dataset $DATASET"
    python ./EG_MCTS.py --start $CURRENT_START --end $CURRENT_END --ss_model $SS_MODEL --dataset $DATASET &
    
    # Update for the next iteration
    CURRENT_START=$((CURRENT_END))
done

# Wait for all background jobs to finish
echo "Waiting for all jobs to complete..."
wait

echo "========================================================"
echo "All jobs completed successfully."
echo "========================================================"