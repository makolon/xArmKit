#!/bin/bash
# Build and run xArmKit Docker container

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building xArmKit Docker image...${NC}"
docker-compose build

echo -e "${GREEN}Starting xArmKit container...${NC}"

# Set DISPLAY if not already set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo -e "${YELLOW}DISPLAY not set, using :0${NC}"
fi

# Allow X11 forwarding
xhost +local:docker 2>/dev/null || echo -e "${YELLOW}Warning: xhost command failed, GUI may not work${NC}"

# Set X11 socket permissions
if [ -d "/tmp/.X11-unix" ]; then
    sudo chmod 1777 /tmp/.X11-unix 2>/dev/null || true
fi

# Run container
docker-compose up -d

echo -e "${GREEN}Container started!${NC}"
echo -e "${YELLOW}To enter the container, run:${NC}"
echo -e "  docker exec -it xarmkit_dev /bin/bash"
echo -e ""
echo -e "${YELLOW}To run object detection (headless mode):${NC}"
echo -e "  docker exec -it xarmkit_dev python src/perception/detect_bbox.py --objects cup bottle --headless --output /workspace/detection_output.png"
echo -e ""
echo -e "${YELLOW}To run with GUI (requires X11):${NC}"
echo -e "  docker exec -it xarmkit_dev python src/perception/detect_bbox.py --objects cup bottle"
