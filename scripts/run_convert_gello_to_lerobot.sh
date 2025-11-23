#!/bin/bash

# Convert Gello PKL datasets to LeRobot v3.0 format
# This script wraps the conversion process with default parameters

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default parameters
INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/datasets/gello_data/gello}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/datasets/lerobot_gello}"
FPS="${FPS:-10}"
ROBOT_TYPE="${ROBOT_TYPE:-xarm}"
TASK_NAME="${TASK_NAME:-}"
GOAL="${GOAL:-}"
OVERWRITE="${OVERWRITE:-true}"

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Gello to LeRobot Converter${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Input directory:  ${INPUT_DIR}"
echo -e "Output directory: ${OUTPUT_DIR}"
echo -e "FPS:              ${FPS}"
echo -e "Robot type:       ${ROBOT_TYPE}"
echo -e "Task name:        ${TASK_NAME:-(auto-detect)}"
echo -e "Goal:             ${GOAL:-(auto-generate)}"
echo -e "Overwrite:        ${OVERWRITE}"
echo -e "${GREEN}========================================${NC}\n"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory not found: $INPUT_DIR${NC}"
    echo -e "${YELLOW}Please set INPUT_DIR environment variable or create the directory${NC}"
    exit 1
fi

# Build command arguments
CMD_ARGS=(
    --input "$INPUT_DIR"
    --output "$OUTPUT_DIR"
    --fps "$FPS"
    --robot-type "$ROBOT_TYPE"
)

# Add optional arguments if provided
if [ -n "$TASK_NAME" ]; then
    CMD_ARGS+=(--task-name "$TASK_NAME")
fi

if [ -n "$GOAL" ]; then
    CMD_ARGS+=(--goal "$GOAL")
fi

# Add overwrite flag if enabled
if [ "$OVERWRITE" = "true" ]; then
    CMD_ARGS+=(--overwrite)
fi

# Run the conversion script
echo -e "${GREEN}Starting conversion...${NC}\n"
python "$PROJECT_ROOT/src/datasets/conversion/convert_gello_to_lerobot.py" "${CMD_ARGS[@]}"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Conversion completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "Output saved to: ${OUTPUT_DIR}"
    
    # Show dataset info if lerobot is available
    if command -v python &> /dev/null; then
        echo -e "\n${YELLOW}Verifying dataset...${NC}"
        python -c "
from pathlib import Path
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('$OUTPUT_DIR')
    print(f'✓ Dataset loaded successfully')
    print(f'  Episodes: {ds.num_episodes}')
    print(f'  Frames: {ds.num_frames}')
    print(f'  FPS: {ds.fps}')
except Exception as e:
    print(f'Note: Could not verify dataset - {e}')
" 2>/dev/null || echo -e "${YELLOW}(Dataset verification skipped - lerobot not available)${NC}"
    fi
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}Conversion failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
