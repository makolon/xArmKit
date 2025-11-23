#!/bin/bash

# Data collection script
# This script sequentially starts:
# 1. xArm robot server
# 2. RealSense camera server
# 3. Gello data collection

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Array to store process IDs
PIDS=()

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down all processes...${NC}"
    for pid in "${PIDS[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing process $pid"
            kill -TERM $pid 2>/dev/null || true
        fi
    done
    wait
    echo -e "${GREEN}All processes stopped.${NC}"
    exit 0
}

# Cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Set path to gello_software
GELLO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../third_party/gello_software" && pwd)"
cd "$GELLO_DIR"

echo -e "${GREEN}Starting data collection pipeline...${NC}"
echo -e "Working directory: ${GELLO_DIR}"

# Set data directory
DATA_DIR="${DATA_DIR:-./datasets/gello_data}"
echo -e "Data will be saved to: ${DATA_DIR}\n"

# 1. Start xArm robot server
echo -e "${GREEN}[1/3] Starting xArm robot server...${NC}"
python experiments/launch_nodes.py --robot=xarm &
ROBOT_PID=$!
PIDS+=($ROBOT_PID)
echo -e "Robot server started (PID: $ROBOT_PID)"
sleep 3  # Wait for robot server to start

# 2. Start RealSense camera server
echo -e "${GREEN}[2/3] Starting RealSense camera server...${NC}"
python experiments/launch_camera_nodes.py &
CAMERA_PID=$!
PIDS+=($CAMERA_PID)
echo -e "Camera server started (PID: $CAMERA_PID)"
sleep 3  # Wait for camera server to start

# 3. Start Gello data collection (foreground)
echo -e "${GREEN}[3/3] Starting Gello data collection...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop data collection${NC}\n"
python experiments/run_env.py --agent=gello --use-save-interface --data-dir="$DATA_DIR"

# Cleanup when data collection ends
cleanup
