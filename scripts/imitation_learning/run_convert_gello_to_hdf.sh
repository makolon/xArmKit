#!/bin/bash
#
# Convert Gello PKL datasets to HDF5 format with optional annotations
#
# Usage:
#   ./run_convert_gello_to_hdf.sh
#
# Options for annotation:
#   - Set DETECT_BBOX=true to detect bounding boxes
#   - Set DETECT_MASK=true to detect segmentation masks
#   - Set ESTIMATE_POSE=true to estimate 6D poses
#   - Set OBJECTS to list of object names (space-separated)
#

# ============================================================================
# Configuration
# ============================================================================

# Input/Output paths
INPUT_DIR="${INPUT_DIR:-./datasets/gello_data/gello}"
OUTPUT_FILE="${OUTPUT_FILE:-./datasets/gello_hdf5/data.h5}"

# Object names to detect (space-separated)
OBJECTS="${OBJECTS:-}"

# Annotation flags
DETECT_BBOX="${DETECT_BBOX:-false}"
DETECT_MASK="${DETECT_MASK:-false}"
ESTIMATE_POSE="${ESTIMATE_POSE:-false}"

# Model paths (update these to your actual paths)
GROUNDING_DINO_CONFIG="${GROUNDING_DINO_CONFIG:-./third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDING_DINO_CHECKPOINT="${GROUNDING_DINO_CHECKPOINT:-./checkpoints/groundingdino_swint_ogc.pth}"

SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-./checkpoints/sam2.1_hiera_large.pt}"
SAM2_CONFIG="${SAM2_CONFIG:-./third_party/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"

MESH_DIR="${MESH_DIR:-./data/meshes}"

FOUNDATIONPOSE_SCORE_CKPT="${FOUNDATIONPOSE_SCORE_CKPT:-}"
FOUNDATIONPOSE_REFINE_CKPT="${FOUNDATIONPOSE_REFINE_CKPT:-}"

# Camera parameters
CAMERA_FX="${CAMERA_FX:-605.0}"
CAMERA_FY="${CAMERA_FY:-605.0}"

# Device
DEVICE="${DEVICE:-cuda}"

# ============================================================================
# Script
# ============================================================================

# Navigate to project root
cd "$(dirname "$0")/../.." || exit 1

echo "============================================================================"
echo "Gello PKL -> HDF5 Converter"
echo "============================================================================"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_FILE"
echo "Objects:      ${OBJECTS:-None}"
echo "Detect BBox:  $DETECT_BBOX"
echo "Detect Mask:  $DETECT_MASK"
echo "Estimate Pose: $ESTIMATE_POSE"
echo "Device:       $DEVICE"
echo "============================================================================"
echo ""

# Build command
CMD="python src/datasets/conversion/convert_gello_to_hdf.py \
    --input \"$INPUT_DIR\" \
    --output \"$OUTPUT_FILE\" \
    --device \"$DEVICE\" \
    --camera-fx $CAMERA_FX \
    --camera-fy $CAMERA_FY"

# Add objects if specified
if [ -n "$OBJECTS" ]; then
    CMD="$CMD --objects $OBJECTS"
fi

# Add annotation flags and model paths
if [ "$DETECT_BBOX" = "true" ]; then
    CMD="$CMD --detect-bbox \
        --grounding-dino-config \"$GROUNDING_DINO_CONFIG\" \
        --grounding-dino-checkpoint \"$GROUNDING_DINO_CHECKPOINT\""
fi

if [ "$DETECT_MASK" = "true" ]; then
    CMD="$CMD --detect-mask \
        --sam2-checkpoint \"$SAM2_CHECKPOINT\" \
        --sam2-config \"$SAM2_CONFIG\""
fi

if [ "$ESTIMATE_POSE" = "true" ]; then
    CMD="$CMD --estimate-pose \
        --mesh-dir \"$MESH_DIR\""
    
    if [ -n "$FOUNDATIONPOSE_SCORE_CKPT" ]; then
        CMD="$CMD --foundationpose-score-ckpt \"$FOUNDATIONPOSE_SCORE_CKPT\""
    fi
    
    if [ -n "$FOUNDATIONPOSE_REFINE_CKPT" ]; then
        CMD="$CMD --foundationpose-refine-ckpt \"$FOUNDATIONPOSE_REFINE_CKPT\""
    fi
fi

# Execute
echo "Running command:"
echo "$CMD"
echo ""

eval "$CMD"

echo ""
echo "============================================================================"
echo "Conversion complete!"
echo "============================================================================"
