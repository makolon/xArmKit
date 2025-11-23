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

# Set paths
XARMKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GELLO_DIR="${XARMKIT_DIR}/third_party/gello_software"

# Kill any existing processes from previous runs
echo -e "${YELLOW}Cleaning up any existing processes...${NC}"
pkill -f "launch_camera_nodes.py" 2>/dev/null || true
pkill -f "launch_nodes.py" 2>/dev/null || true
sleep 1  # Wait for processes to fully terminate

echo -e "${GREEN}Starting data collection pipeline...${NC}"

# 0. Reset xArm7 to home position
echo -e "${GREEN}[0/3] Resetting xArm7 to home position...${NC}"
cd "$XARMKIT_DIR"
python -c "
import sys
sys.path.insert(0, 'src')
from real.xarm7 import XArm7

robot_ip = '192.168.10.211'
print(f'Connecting to xArm7 at {robot_ip}...')
with XArm7(robot_ip, is_radian=False, enable_logging=True) as robot:
    if robot.initialize(go_home=True):
        print('xArm7 successfully moved to home position!')
    else:
        print('Failed to initialize xArm7!')
        sys.exit(1)
"
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to reset xArm7 to home position${NC}"
    exit 1
fi
echo -e "${GREEN}xArm7 reset complete${NC}\n"

cd "$GELLO_DIR"
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
